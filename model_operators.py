import inspect, ast, astor
from ast import *

class ChangeFunctionName(ast.NodeTransformer):
    def __init__(self, new_name):
        self.new_name = new_name
        super().__init__()
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        newnode = node
        newnode.name = self.new_name
        return newnode

#class AddToFunctionBody(ast.NodeTransformer):
#    """
#    Adds code to either the beginning (default) or end of a function
#    """
#    def __init__(self, code, head=True, skip_lines = 0):
#        self.code = code
#        self.head = head
#        self.skip_lines = skip_lines
#        super().__init__()
#    def visit_FunctionDef(self, node):
#        self.generic_visit(node)
#        if self.head:
#            #insert the whole code statement first in the body of the function
#            #node.body = [self.code] + node.body
#            node.body = node.body[:self.skip_lines] + [self.code] + node.body[self.skip_lines:]
#        else:
#            #append the whole code statement last in the body of the function
#            node.body.append(self.code)
#            # switch code and return statement so it's last
#            node.body[-1], node.body[-2] = node.body[-2], node.body[-1]
#        return node

class AddToFunctionBody(ast.NodeTransformer):
    """
    Adds code to either the beginning (default) or end of a function
    """
    def __init__(self, code, pos=0):
        self.code = code
        self.pos = pos
        super().__init__()
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if not type(self.code) == list:
            self.code = [self.code]
        for line in self.code:
            node.body = node.body[:self.pos] + [line] + node.body[self.pos:]
            if self.pos != -1:
                self.pos += 1
        return node

class AddToPlate(ast.NodeTransformer):
    """
    Identifies plate with specified name, and appends specified code to its body. 
    If code is None, assigns plate to self.plate and deletes it
    """
    def __init__(self, plate_name, code, pos=0):
        self.plate_name = plate_name
        self.code = code
        self.pos = pos
        super().__init__()
    def visit_With(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        withexpr = node.items[0].context_expr
        # assumes that plate names are on the form f'K_{_id}'
        if withexpr.func.attr == 'plate' and withexpr.args[0].values[0].s == self.plate_name+'_':
            self.plate = node
            if self.code is None:
                return
            else:
                if not type(self.code) == list:
                    self.code = [self.code]
                for line in self.code:
                    node.body = node.body[:self.pos] + [line] + node.body[self.pos:]
                    if self.pos != -1:
                        self.pos += 1
                return node
        else:
            return node

class GetPlateIndex(ast.NodeTransformer):
    """
    Get the index of the sought plate in the body-list of whatever ast object (function or with.plate) contains it.
    """
    def __init__(self, plate_name):
        self.plate_name = plate_name
        self.pos = -1
        self.container = None

    def visit_With(self, node):
        self.generic_visit(node)
        withexpr = node.items[0].context_expr
        # assumes that plate names are on the form f'K_{_id}'
        #print(f"Looking in plate {(withexpr.args[0].values[0].s[:-1])}")
        if withexpr.func.attr == 'plate':
            for i,elem in enumerate(node.body):
          #      if type(elem) == ast.With:
         #           print(f'Found plate {elem.items[0].context_expr.args[0].values[0].s}')
                if (type(elem) == ast.With and elem.items[0].context_expr.args[0].values[0].s == self.plate_name+'_'):
                    self.pos = i
                    self.container = withexpr.args[0].values[0].s[:-1]
            return node
        else:
            return node
        
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        for i,elem in enumerate(node.body):
            if (type(elem) == ast.With and elem.items[0].context_expr.args[0].values[0].s == self.plate_name+'_'):
                self.pos = i
                self.container = ''
        return node

class AddToForLoop(ast.NodeTransformer):
    """
    Identifies for loop with specified indexing variable, and appends specified code to its body. 
    """
    def __init__(self, target_indexing_variable, code):
        self.target_indexing_variable = target_indexing_variable
        self.code = code
        super().__init__()
    def visit_For(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        indexing_variable = node.target.id
        if indexing_variable == self.target_indexing_variable:
            if self.code is None:
                newnode = node
                newnode.body = []
                return newnode
            else:
                newnode = node
                newnode.body.append(self.code)
                return newnode
        else:
            return node

class AddReturn(ast.NodeTransformer):
    """
    Takes a single or a list of AST objects and adds a corresponding return statement to the tree 
    """
    def __init__(self, return_tuple):
        # if there are several ast objects to be returned
        if hasattr(return_tuple, '__iter__'):
            self.return_tuple = Return(value=Tuple(elts=[Name(id=elem,ctx=Load()) for elem in return_tuple],ctx=Load()))
        # otherwise there's just one
        else:
            self.return_tuple = Return(value=return_tuple)
        super().__init__()
    def visit_FunctionDef(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        node.body.append(self.return_tuple)
        return node

class AddArgsToFunctionDef(ast.NodeTransformer):
    """
    Adds arguments to either the end (default) or beginning of a function definition
    """
    def __init__(self, arg_to_add, pos = 0):
        self.pos = pos
        self.arg_to_add = arg_to_add
        super().__init__()
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        new_arg = arg(arg=self.arg_to_add, annotation=None)
        node.args.args = node.args.args[:self.pos] + [new_arg] + node.args.args[self.pos:]
        return node

class DeletePlate(ast.NodeTransformer):
    """
    Identifies plate with specified name, and appends specified code to its body. 
    If code is None, assigns plate to self.plate and deletes it
    """
    def __init__(self, plate_name, code):
        self.plate_name = plate_name
        super().__init__()
    def visit_With(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        withexpr = node.items[0].context_expr
        # assumes that plate names are on the form f'K_{_id}'
        if withexpr.func.attr == 'plate' and withexpr.args[0].values[0].s.startswith(self.plate_name):
            self.code = node
            return
        else:
            return node

class CutFromPlate(ast.NodeTransformer):
    """
    Identifies plate with specified name, and cuts first or last statement form its body
    If code is None, assigns plate to self.plate and deletes it
    """
    def __init__(self, plate_name, head = True):
        self.plate_name = plate_name
        self.head = head
        super().__init__()
    def visit_With(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        withexpr = node.items[0].context_expr
        # assumes that plate names are on the form f'K_{_id}'
        if withexpr.func.attr == 'plate' and withexpr.args[0].values[0].s.startswith(self.plate_name):
            if self.head:
                #cut the first element in body
                self.code = node.body[0]
                node.body = node.body[1:]
            else:
                #cut the last element in body
                self.code = node.body[-1]
                node.body = node.body[:-1]
        return node

class CutArgsFromFunctionDef(ast.NodeTransformer):
    """
    Cuts arguments from either the beginning (default) or end of a function definition
    """
    def __init__(self, n_args_to_cut = 1, head = True):
        self.n_args_to_cut = n_args_to_cut
        self.head = head
        super().__init__()
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.head:
            node.args.args = node.args.args[self.n_args_to_cut:]
        else:
            node.args.args = node.args.args[:-self.n_args_to_cut]
        return node


class CutFromFunctionBody(ast.NodeTransformer):
    """
    Cuts code from either the beginning (default) or end of a function
    """
    def __init__(self, head=True):
        self.head = head
        super().__init__()
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self.head:
            #cut the first element in body
            self.code = node.body[0]
            node.body = node.body[1:]
        else:
            #cut the last element in body
            self.code = node.body[-1]
            node.body = node.body[:-1]
        return node

class GetObservationModel(ast.NodeTransformer):
    """
    Identifies sampling site with name 'obs', and returns its parameters
    """
    def __init__(self):
        super().__init__()
    def visit_Call(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        if node.func.attr == 'sample' and node.args[0].s == 'obs':
            self.code = node
            #self.code = node.args[1] 
            return node
        else:
            return node

class GetNames(ast.NodeTransformer):
    """
    Returns all ast Names
    This includes things like torch from 'torch.zeros(D)'
    which can be avoided by naming it in the model, e.g. 'loc = torch.zeros(D)'
    and using loc in the observation model
    """
    def __init__(self):
        super().__init__()
        self.code = []
    def visit_Name(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        self.code.append(node)
        return node

class GetDistributionParameters(ast.NodeTransformer):
    """
    Finds a call to dist.<something> and adds all its parameters to self.code
    """
    def __init__(self):
        super().__init__()
        self.code = []
    def visit_Call(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        if node.func.value.id == 'dist':
            for arg in node.args:
                self.code.append(arg.id)
            for keyword in node.keywords:
                self.code.append(keyword.value.id)
        return node

class ChangeObservationModel(ast.NodeTransformer):
    """
    Identifies sampling site with name 'obs', and replaces its distribution with a specified one
    """
    def __init__(self, new_obs_model):
        self.new_obs_model = new_obs_model
        super().__init__()
    def visit_Call(self, node):
        # we want to visit child nodes, so visit it
        self.generic_visit(node)
        if node.func.attr == 'sample' and node.args[0].s == 'obs':
            newnode = node
            newnode.args[1] = self.new_obs_model
            return newnode
        else:
            return node
        
def change_observation_model_to_LowRankMultivariateNormal(tree):
    lowrank_normal_obs_model = ast.Call(func=ast.Attribute(value=ast.Name(id='dst', ctx=ast.Load()), attr='LowRankMultivariateNormal', ctx=ast.Load()),\
                                args=[ast.Name(id='loc', ctx=ast.Load())],\
                                keywords=[ast.keyword(arg='cov_factor', value=ast.Name(id='cov_factor', ctx=ast.Load())),\
                                ast.keyword(arg='cov_diag', value=ast.Name(id='cov_diag', ctx=ast.Load()))]) 
    ChangeObservationModel(lowrank_normal_obs_model).visit(tree)

