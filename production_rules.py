import pyro
import torch
import sys
from ast import *
sys.path.append("..")
from models_and_guides import *
from model_operators import *

class mixtureOf(Model):
    def __init__(self, K, component_model, _id):
        super(mixtureOf, self).__init__(component_model.X, component_model.batch_size, _id)
        self.K = K
        self.component_model = component_model
        self.param_shapes_and_support = self.get_param_shapes_and_support(self._id)
        self.param_init = self.initialize_parameters()
        self.component_param_shapes_and_support = []
        for k in range(self.K):
            # get names and shapes of component params with k as id
            self.component_param_shapes_and_support.append(component_model.get_param_shapes_and_support(k))
            # init them
            #component_priors = set_uninformative_priors(self.component_param_shapes_and_support[k])
            component_priors = get_random_init(self.component_param_shapes_and_support[k])
            # merge with param_init dict of n'th component model
            self.param_init = {**self.param_init, **component_priors}
        # change model to include component, save to a file we import from in the next line
        compileMixture(self)
        exec('from compiled_models_' + self._id + ' import compiled_component_model, compiled_model', globals())
        self.compiled_component_model = compiled_component_model
        self.compiled_model = compiled_model
        compileMixtureGuide(self)
        exec('from compiled_guides_' + self._id + ' import compiled_component_guide, compiled_guide', globals())
        self.compiled_component_guide = compiled_component_guide
        self.compiled_guide = compiled_guide
    
    def get_param_shapes_and_support(self, _id = None):
        if _id == None:
            _id = self._id
        return {f'mixing_proportions_concentration_{_id}': ((self.K,), constraints.positive)}
    
    def model_prototype(self, X):
        mixing_proportions_concentration = self.param_init[f'mixing_proportions_concentration_{self._id}']
        mixing_proportions = pyro.sample(f'mixing_proportions_{self._id}', dist.Dirichlet(mixing_proportions_concentration))
        # add one list per parameter in component's obs model
        #locs = torch.empty((self.K,self.D))
        #cov_factors = torch.empty((self.K,self.D,self.component_model.K))
        #cov_diags = torch.empty((self.K,self.D))
        for k in pyro.plate(f'K_{self._id}', self.K):
            pass
            # call component model, append returned parameters to lists
            #locs[k], cov_factors[k], cov_diags[k] = compiled_component_model(self,k)
        with pyro.plate(f'N_{self._id}', size=N, subsample_size=self.batch_size) as ind:
            assignment = pyro.sample(f'assignment_{self._id}', dist.Categorical(mixing_proportions), infer={"enumerate": "parallel"})
            # add X = component obs model, where parameters are indexed by assignment
        return X

    def guide_prototype(self, X):
        mixing_proportions_concentration = self.param_init[f'mixing_proportions_concentration_{self._id}']
        mixing_proportions = pyro.sample(f'mixing_proportions_{self._id}', dist.Dirichlet(mixing_proportions_concentration))
        for k in pyro.plate(f'K_{self._id}', self.K):
            pass
        with pyro.plate(f'N_{self._id}', size=N, subsample_size=self.batch_size) as ind:
            assignment = pyro.sample(f'assignment_{self._id}', dist.Categorical(mixing_proportions), infer={"enumerate": "parallel"})

    def model(self, X):
        return self.compiled_model(self, X)

    def guide(self, X):
        return self.compiled_guide(self, X)

def compileComponentModel(component):
    """
    Remove observation model from a model and save the resulting latent component model to a file
    Returns the obsmodel parameters, and obsmodel code
    """
    # get the source code of model function, stripping whitespace
    if type(component) == mixtureOf:
        source = inspect.getsource(component.compiled_model).strip()
    else:
        source = inspect.getsource(component.model).strip()
    tree = parse(source)
    new_name = 'compiled_component_model'
    change_name = ChangeFunctionName(new_name).visit(tree)
    # remove X argument from function definition
    CutArgsFromFunctionDef(n_args_to_cut=1, head=False).visit(tree)
    # add _id argument to function definition
    AddArgsToFunctionDef('_id').visit(tree)
    # define the remove a line operator (call as many times as the number of lines to be removed)
    cutline = CutFromFunctionBody()
    # remove _id = self._id
    # remove K = self.K
    # remove N, D = X.shape
    cutline.visit(tree)
    cutline.visit(tree)
    cutline.visit(tree)
    # add K assigment
    AddToFunctionBody(Assign(targets=[Name(id='K')], value=Num(n=component.K))).visit(tree)
    # add D assigment
    AddToFunctionBody(Assign(targets=[Name(id='D')], value=Num(n=component.D))).visit(tree)
    # delete plate with observation model, save it to delete.code
    delete = DeletePlate('N', code=None)
    delete.visit(tree)
    # isolate observation model
    obsmodel = GetObservationModel()
    obsmodel.visit(delete.code)
    # get its parameters
    getnames = GetDistributionParameters()
    getnames.visit(obsmodel.code)
    # cut last line from function body, i.e. the return statement
    CutFromFunctionBody(head=False).visit(tree) 
    # make new return statement (first name in params is 'dist', so we omit it)
    # add new return statement
    AddReturn(getnames.code).visit(tree)
    fix_missing_locations(tree)
    new_source = astor.to_source(ast.parse(tree))
    return new_source, getnames.code, obsmodel.code

def compileComponentGuide(component):
    """
    """
    # get the source code of model function, stripping whitespace
    source = inspect.getsource(component.guide).strip()
    tree = parse(source)
    new_name = 'compiled_component_guide'
    change_name = ChangeFunctionName(new_name)
    change_name.visit(tree)
    # remove X argument from function definition
    CutArgsFromFunctionDef(n_args_to_cut=1, head=False).visit(tree)
    # add _id argument to function definition
    AddArgsToFunctionDef('_id').visit(tree)
    # define the remove a line operator (call as many times as the number of lines to be removed)
    cutline = CutFromFunctionBody()
    # remove _id = self._id
    cutline.visit(tree)
    # remove K = self.K
    cutline.visit(tree)
    # remove N, D = X.shape
    cutline.visit(tree)
    # add K assigment
    AddToFunctionBody(Assign(targets=[Name(id='K')], value=Num(n=component.K))).visit(tree)
    # add D assigment
    AddToFunctionBody(Assign(targets=[Name(id='D')], value=Num(n=component.D))).visit(tree)
    fix_missing_locations(tree)
    new_source = astor.to_source(ast.parse(tree))
    return new_source

def compileMixture(mixture):
    """
    Compile a mixture model
    # TODO:
    """
    D = mixture.D
    # change component model to latent model, save to compiled_code.py
    new_source, obsparams, obsmodelcode = compileComponentModel(mixture.component_model)
    with open("compiled_models_"+mixture._id+".py", "w") as output:
        output.write('from models_and_guides import *\n\n')
        output.writelines(new_source)
    mixture.obsparams = obsparams
    # add an empty tensor for every parameter called by component's obs model, 
    # with shape given by component model instance
    # assumes all component models have same parameter shapes
    componentModelTrace = pyro.poutine.trace(mixture.component_model.model).get_trace(mixture.component_model.X)
    # get shapes of parameters of observation dist of component model, ignoring batch dimension
    param_shapes = [(mixture.K,) + getattr(componentModelTrace.nodes['obs']['fn'],param).shape[1:] for param in obsparams]
    # translate to AST
    param_shapes_code = [[Num(n=numeral) for numeral in param_shape] for param_shape in param_shapes]
    # construct empty tensor assign statements
    param_list_statements = [Assign(targets=[Name(id=param + 's', ctx=Store())], value=Call(func=Attribute(value=Name(id='torch', ctx=Load()), attr='empty', ctx=Load()),
                        args=[Tuple(elts=shape, ctx=Load())],
                        keywords=[])) for shape,param in zip(param_shapes_code,obsparams)]
    source = inspect.getsource(mixture.model_prototype).strip()
    tree = parse(source)
    #new_name = 'mixtureOf' + model.model.__name__ + 's'
    new_name = 'compiled_model'
    change_name = ChangeFunctionName(new_name)
    change_name.visit(tree)

    # data shape must be inferred from input
    # so that we can reuse model for MNLL calculation
    get_data_shape = Assign(targets=[Tuple(elts=[Name(id='N'), Name(id='D')])], value=Attribute(value=Name(id='X'), attr='shape'))
    AddToFunctionBody(get_data_shape).visit(tree)

    for param_list_statement in param_list_statements:
        add_param_list = AddToFunctionBody(param_list_statement)
        add_param_list.visit(tree)
    # call component model, append returned parameters to lists
    elts = [Subscript(value=Name(id=param + 's', ctx=Load()), slice=Index(value=Name(id='k', ctx=Load())), ctx=Store()) for param in obsparams]
    component_call = Assign(targets=[Tuple(elts=elts, ctx=Store())],
                        value=Call(func=Attribute(value=Name(id='self'), attr='compiled_component_model'),args=[Name(id='self'), Name(id='k')],keywords=[]))
                        #value=Call(func=Name(id='self.compiled_component_model', ctx=Load()), args=[Name(id='k', ctx=Load())], keywords=[]))
    add_component_call = AddToForLoop('k', component_call)
    add_component_call.visit(tree)

    # add indexing by assignment latent variable in new obs model
    newobsmodel_args = [Subscript(value=Name(id=param+'s', ctx=Load()), slice=Index(value=Name(id='assignment', ctx=Load())), ctx=Load()) for param in obsparams]
    # remove keyword args in the obsmodel
    obsmodelcode.args[1].keywords = []
    # add indexed args to obsmodel
    obsmodelcode.args[1].args = newobsmodel_args
    newobsmodel = Assign(targets=[Name(id='X', ctx=Store())], value=obsmodelcode)
    # add new obs model
    add_new_obs_model = AddToPlate('N', newobsmodel)
    add_new_obs_model.visit(tree)
    fix_missing_locations(tree)
    new_source = astor.to_source(ast.parse(tree))
    with open("compiled_models_"+mixture._id+".py", "a") as output:
        output.write("\n")
        output.writelines(new_source)

def compileMixtureGuide(mixture):
    """
    Assumes the model has been compiled, and uses it as a template for the guide.
    """
    new_source = compileComponentGuide(mixture.component_model)
    with open("compiled_guides_"+mixture._id+".py", "w") as output:
        output.write('from models_and_guides import *\n\n')
        output.writelines(new_source)
    source = inspect.getsource(mixture.compiled_model)
    tree = parse(source)
    new_name = 'compiled_guide'
    change_name = ChangeFunctionName(new_name)
    change_name.visit(tree)

    # delete plate with observation model, save it to delete.code
    delete = DeletePlate('N', code=None)
    delete.visit(tree)
    # make new return statement (first name in params is 'dist', so we omit it)
    params = ['mixing_proportions'] + [obsparam + 's' for obsparam in mixture.obsparams]
    # cut last line from function body, i.e. the return statement
    CutFromFunctionBody(head=False).visit(tree) 
    # add new return statement
    AddReturn(params).visit(tree)
    fix_missing_locations(tree)
    new_source = astor.to_source(ast.parse(tree))
    with open("compiled_guides_"+mixture._id+".py", "a") as output:
        output.write("\n")
        output.writelines(new_source)