# compositions_mixtures_factors
Research prototype for Pyro code generation and compositional model search.

There are three main functionalities:
1. Given a probabilistic graphical models (PGM) defined as a graph, apply a subgraph substitution, returning a new graphical model. 
2. Compile a PGM to a Pyro model, and train it with stochastic variational inference.
3. Given a trained "teacher" graphical model A, and an untrained "student" model B derived from A, initialize B using the parameters of A. This usually gives orders of magnitude faster inference on B.

*Random\ splits.ipynb* demonstrates 1., 2. and 3. for model selection on synthetic data generated from a mixture of factor analyses.

*model_operators.py* defines AST operators for modifying parts of Pyro models and guides

*code_generation.py* contains the graph to Pyro model compiler

*graph_grammar.py* implements several subgraph substitutions
  * Marginalizing the local latent variables out in a factor analysis PGM
  * Creating a mixture of factor analyses from a factor analysis PGM

*inference.py* implements mini-batch stochastic variational inference for compiled PGMs, and includes
  * Convergence estimation by linear regression on ELBO
  * Tracking of gradient norms and parameter values during training
  * Tracking of mean held-out predictive likelihood on a test set
  * Checkpoint saving
  * The successive halving algorithm for hyperparameter tuning

*initializations.py* defines initializers for compiled models, including
  * random hyperparameter iniitalization and weakly informative priors
  * given a teacher and student model, initialize the student

*models_and_guides.py* contains the main model class, which includes a number of convenience functions, and various models in the mixture/factor family

*tracepredictive.py* implements a variant of Pyro's tracepredictive which was buggy for some models at the time of writing

Completing the prototype requires
1. Adding subgraph substitutions to *graph_grammar.py*
2. Definiting how models can iniitalize each other
3. Adding a search algorithm such as MCTS using held-out predictive likelihood as the criterion

and would yield a system that can learn to construct and efficiently find and train generative models for any vector-valued dataset.
