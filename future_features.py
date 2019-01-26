from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from contextlib import contextmanager

import functools
import threading
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops.distributions import bijector
from tensorflow_probability.python.internal import distribution_util
ed = tfp.edward2
interception = ed.interception

@contextmanager
def tape():
  """Context manager for recording interceptable executions onto a tape.
  Similar to `tf.GradientTape`, operations are recorded if they are executed
  within this context manager. In addition, the operation must be registered
  (wrapped) as `ed.interceptable`.
  Yields:
    tape: OrderedDict where operations are recorded in sequence. Keys are
      the `name` keyword argument to the operation (typically, a random
      variable's `name`) and values are the corresponding output of the
      operation. If the operation has no name, it is not recorded.
  #### Examples
  ```python
  from tensorflow_probability import edward2 as ed
  def probabilistic_matrix_factorization():
    users = ed.Normal(0., 1., sample_shape=[5000, 128], name="users")
    items = ed.Normal(0., 1., sample_shape=[7500, 128], name="items")
    ratings = ed.Normal(loc=tf.matmul(users, items, transpose_b=True),
                        scale=0.1,
                        name="ratings")
    return ratings
  with ed.tape() as model_tape:
    ratings = probabilistic_matrix_factorization()
  assert model_tape["users"].shape == (5000, 128)
  assert model_tape["items"].shape == (7500, 128)
  assert model_tape["ratings"] == ratings
  ```
  """
  tape_data = collections.OrderedDict({})

  def record(f, *args, **kwargs):
    """Records execution to a tape."""
    name = kwargs.get("name")
    output = f(*args, **kwargs) #modified from output = interceptable(f)(*args, **kwargs) 
    if name:
      tape_data[name] = output
    return output

  with interception(record):
    yield tape_data

class SoftmaxCentered(bijector.Bijector):
  """Bijector which computes `Y = g(X) = exp([X 0]) / sum(exp([X 0]))`.
  To implement [softmax](https://en.wikipedia.org/wiki/Softmax_function) as a
  bijection, the forward transformation appends a value to the input and the
  inverse removes this coordinate. The appended coordinate represents a pivot,
  e.g., `softmax(x) = exp(x-c) / sum(exp(x-c))` where `c` is the implicit last
  coordinate.
  Example Use:
  ```python
  bijector.SoftmaxCentered().forward(tf.log([2, 3, 4]))
  # Result: [0.2, 0.3, 0.4, 0.1]
  # Extra result: 0.1
  bijector.SoftmaxCentered().inverse([0.2, 0.3, 0.4, 0.1])
  # Result: tf.log([2, 3, 4])
  # Extra coordinate removed.
  ```
  At first blush it may seem like the [Invariance of domain](
  https://en.wikipedia.org/wiki/Invariance_of_domain) theorem implies this
  implementation is not a bijection. However, the appended dimension
  makes the (forward) image non-open and the theorem does not directly apply.
  """

  def __init__(self,
                validate_args=False,
                name="softmax_centered"):
    self._graph_parents = []
    self._name = name
    super(SoftmaxCentered, self).__init__(
        forward_min_event_ndims=1,
        validate_args=validate_args,
        name=name)

  def _forward_event_shape(self, input_shape):
    if input_shape.ndims is None or input_shape[-1] is None:
      return input_shape
    return tf.TensorShape([input_shape[-1] + 1])

  def _forward_event_shape_tensor(self, input_shape):
    return (input_shape[-1] + 1)[..., tf.newaxis]

  def _inverse_event_shape(self, output_shape):
    if output_shape.ndims is None or output_shape[-1] is None:
      return output_shape
    if output_shape[-1] <= 1:
      raise ValueError("output_shape[-1] = %d <= 1" % output_shape[-1])
    return tf.TensorShape([*output_shape[:-1],output_shape[-1] - 1])

  def _inverse_event_shape_tensor(self, output_shape):
    if self.validate_args:
      # It is not possible for a negative shape so we need only check <= 1.
      is_greater_one = tf.assert_greater(
          output_shape[-1], 1, message="Need last dimension greater than 1.")
      output_shape = control_flow_ops.with_dependencies(
          [is_greater_one], output_shape)
    return (*output_shape[:-1], output_shape[-1] - 1)[..., tf.newaxis]

  def _forward(self, x):
    # Pad the last dim with a zeros vector. We need this because it lets us
    # infer the scale in the inverse function.
    y = distribution_util.pad(x, axis=-1, back=True)

    # Set shape hints.
    if x.shape.ndims is not None:
      shape = x.shape[:-1].concatenate(x.shape[-1] + 1)
      y.shape.assert_is_compatible_with(shape)
      y.set_shape(shape)

    return tf.nn.softmax(y)

  def _inverse(self, y):
    # To derive the inverse mapping note that:
    #   y[i] = exp(x[i]) / normalization
    # and
    #   y[end] = 1 / normalization.
    # Thus:
    # x[i] = log(exp(x[i])) - log(y[end]) - log(normalization)
    #      = log(exp(x[i])/normalization) - log(y[end])
    #      = log(y[i]) - log(y[end])

    # Do this first to make sure CSE catches that it'll happen again in
    # _inverse_log_det_jacobian.
    x = tf.log(y)

    log_normalization = (-x[..., -1])[..., tf.newaxis]
    x = x[..., :-1] + log_normalization

    # Set shape hints.
    if y.shape.ndims is not None:
      shape = y.shape[:-1].concatenate(y.shape[-1] - 1)
      x.shape.assert_is_compatible_with(shape)
      x.set_shape(shape)

    return x

  def _inverse_log_det_jacobian(self, y):
    # WLOG, consider the vector case:
    #   x = log(y[:-1]) - log(y[-1])
    # where,
    #   y[-1] = 1 - sum(y[:-1]).
    # We have:
    #   det{ dX/dY } = det{ diag(1 ./ y[:-1]) + 1 / y[-1] }
    #                = det{ inv{ diag(y[:-1]) - y[:-1]' y[:-1] } }   (1)
    #                = 1 / det{ diag(y[:-1]) - y[:-1]' y[:-1] }
    #                = 1 / { (1 + y[:-1]' inv(diag(y[:-1])) y[:-1]) *
    #                        det(diag(y[:-1])) }                     (2)
    #                = 1 / { y[-1] prod(y[:-1]) }
    #                = 1 / prod(y)
    # (1) - https://en.wikipedia.org/wiki/Sherman%E2%80%93Morrison_formula
    #       or by noting that det{ dX/dY } = 1 / det{ dY/dX } from Bijector
    #       docstring "Tip".
    # (2) - https://en.wikipedia.org/wiki/Matrix_determinant_lemma
    return -tf.reduce_sum(tf.log(y), axis=-1)

  def _forward_log_det_jacobian(self, x):
    # This code is similar to tf.nn.log_softmax but different because we have
    # an implicit zero column to handle. I.e., instead of:
    #   reduce_sum(logits - reduce_sum(exp(logits), dim))
    # we must do:
    #   log_normalization = 1 + reduce_sum(exp(logits))
    #   -log_normalization + reduce_sum(logits - log_normalization)
    log_normalization = tf.nn.softplus(
        tf.reduce_logsumexp(x, axis=-1, keep_dims=True))
    return tf.squeeze(
        (-log_normalization + tf.reduce_sum(
            x - log_normalization, axis=-1, keepdims=True)),
        axis=-1)