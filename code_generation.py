import networkx as nx
from model_operators import *
from models_and_guides import DAGModel

def remove_nodes_keep_edges(G, nodes):
    for node in nodes:
        # connect orphaned children to grandparents
        parents = [u for u,v in G.in_edges(node)]
        children = [v for u,v in G.out_edges(node)]
        G.add_edges_from([(u,v) for u,v in zip(parents,children)])
        G.remove_node(node)

def make_plate_graph(DAG):
    # make dependency graph of all plates
    plate_graph = DAG.copy()
    non_plated = [node for node in plate_graph.nodes if 'plates' not in plate_graph.nodes[node].keys()]
    remove_nodes_keep_edges(plate_graph, non_plated)
    for u,v in plate_graph.in_edges:
        try:
            if plate_graph.nodes[v]['plates'] == plate_graph.nodes[u]['plates']:
                plate_graph = nx.algorithms.minors.contracted_nodes(plate_graph,u,v, self_loops=False)
        except KeyError:
            continue
    plate_graph = nx.relabel_nodes(plate_graph, {node:''.join(plate_graph.nodes[node]['plates']) for node in plate_graph.nodes})
    
    # keep merging plates as long as possible
    is_isomorphic = False
    while not is_isomorphic:
        old_plate_graph = plate_graph.copy()
        nodes_to_remove = []
        for node in plate_graph.nodes:
            nodes_to_merge = [(node,other_node) for other_node in plate_graph.nodes if node != other_node and (node.startswith(other_node))]
            for node_pair in nodes_to_merge:
                nodes_to_remove.append(node_pair)
        nodes_to_remove = sorted(nodes_to_remove, reverse=True)
        if nodes_to_remove:
            u,v = nodes_to_remove[0]
            plate_graph = nx.algorithms.minors.contracted_nodes(plate_graph,u,v, self_loops=False)
        is_isomorphic = nx.algorithms.isomorphism.is_isomorphic(old_plate_graph, plate_graph)
    
    # remove self loops
    plate_graph.remove_edges_from(plate_graph.selfloop_edges())
    return plate_graph

def construct_initalization(graph, node):
    if graph.nodes[node]['type'] == 'param':
        return Assign(targets=[Name(id=node+'_init')],
            value=Subscript(value=Attribute(value=Name(id='self'), attr='param_init'),
                slice=Index(
                    value=JoinedStr(
                        values=[Str(s=node+'_init_'),
                            FormattedValue(value=Name(id='_id'), conversion=-1, format_spec=None)]))))

def construct_param(graph, node):
    # need to infer constraint
    if 'constraint' not in graph.nodes[node].keys():
        constraint = []
    else:
        constraint = [keyword(arg='constraint', value=Attribute(value=Name(id='constraints'), attr=graph.nodes[node]['constraint']))]
    return Assign(targets=[Name(id=node)],
            value=Call(func=Attribute(value=Name(id='pyro'), attr='param'),
                args=[
                    JoinedStr(
                        values=[Str(s=node+'_'),
                            FormattedValue(value=Name(id='_id'), conversion=-1, format_spec=None)]),
                    Name(id=node+'_init')],
                keywords=constraint))

def construct_constant(graph, node):
    return Assign(targets=[Name(id=node)], value=Num(n=graph.nodes[node]['value']))

def construct_sample(graph, node):
    distribution = graph.nodes[node]['distribution']
    
    # go through the args of the distribution, find matching ingoing node
    params_in_graph = [graph.edges[edge]['param'] for edge in graph.in_edges(node)]
    params_in_dist = distribution.arg_constraints.keys()
    params = [param for param in params_in_dist if param in params_in_graph]
    args = []
    for param in params:
        arg = [edge[0] for edge in graph.in_edges(node) if graph.edges[edge]['param'] == param][0]
        args.append(Name(id=arg))
    
    # construct distribution with ingoing nodes as arguments
    dist = ''.join(c for c in str(distribution).split('.')[-1] if c.isalnum())
    dist = Call(func=Attribute(value=Name(id='dist'), attr=dist),
                            args=args,
                            keywords=[])
    if graph.nodes[node]['type'] == 'latent':
        node_name = JoinedStr(
                            values=[Str(s=node+'_'),
                                FormattedValue(value=Name(id='_id'), conversion=-1, format_spec=None)])
        keywords = []
    elif graph.nodes[node]['type'] == 'obs':
        node_name = Str(s='obs')
        # assuming this is wrapped in a with pyro.plate(subsampling) as ind:
        keywords = [keyword(arg='obs',
                            value=Call(func=Attribute(value=Name(id='X'), attr='index_select'),
                                args=[Num(n=0), Name(id='ind')],
                                keywords=[]))]
    return Assign(targets=[Name(id=node)],
                value=Call(func=Attribute(value=Name(id='pyro'), attr='sample'),
                    args=[node_name,
                        dist],
                    keywords=keywords))
            

def construct_function(graph, node):
    # assuming function is a torch method, s.t. calling it looks like torch.method(arg1,arg2,...,argN)
    # get string repr of function
    function = graph.nodes[node]['function'].__name__
    args = []
    for edge in graph.in_edges(node):
        assert graph.edges[edge]['type'] == 'arg'
        args.append(Name(id=edge[0]))
    if 'args' in graph.nodes[node]:
        for arg in graph.nodes[node]['args']:
            assert isinstance(arg, ast.AST), f"arg attribute {arg} in node {node} is not an AST object."
            args.append(arg)
    # wrap function call in a deterministic sample site
    if graph.nodes[node]['type'] == 'deterministic':
        return Assign(targets=[Name(id=node)],
            value=Call(func=Attribute(value=Name(id='pyro'), attr='deterministic'),
                args=[
                    JoinedStr(
                            values=[Str(s=node+'_'),
                                FormattedValue(value=Name(id='_id'), conversion=-1, format_spec=None)]),
                    Call(func=Attribute(value=Name(id='torch'), attr=function),
                        args=args,
                        keywords=[])],
                keywords=[]))
    # just write function call as is
    elif graph.nodes[node]['type'] == 'function':
        return Assign(targets=[Name(id=node)],
            value=Call(func=Attribute(value=Name(id='torch'), attr=function),
                        args=args,
                        keywords=[]))

def construct_plate(graph, plate):
    if plate == 'N':
        keywords = [keyword(arg='subsample_size', value=Attribute(value=Name(id='self'), attr='batch_size'))]
        optional_vars=Name(id='ind')
    else: 
        keywords = []
        optional_vars=None
    return With(
            items=[withitem(
                    context_expr=Call(func=Attribute(value=Name(id='pyro'), attr='plate'),
                        args=[
                            JoinedStr(
                                values=[Str(s=plate+'_'),
                                    FormattedValue(value=Name(id='_id'), conversion=-1, format_spec=None)]),
                            Name(id=plate[-1])],
                        keywords=keywords),
                    optional_vars=optional_vars)
                ],
            body=[Pass])

def insert_function_into_class(class_source, function_code, function_def_str, function_return_str):
    start_idx = class_source.find(function_def_str)
    end_idx = class_source.find(function_return_str) + len(function_return_str)
    # add indentation
    function_code = function_code.replace('\n    ','\n        ')
    return class_source.replace(class_source[start_idx:end_idx], function_code)

def generate_model(DAG, dims, root_node_suffix = None):
    # rename root nodes
    if root_node_suffix != None:
        root_node_names = [node for node in DAG.nodes if DAG.in_degree[node] == 0]
        new_root_node_names = ['_'.join(node.split('_')[:-1] + [root_node_suffix] + node.split('_')[-1:]) for node in root_node_names]
        mapping = dict(zip(root_node_names, new_root_node_names))
        DAG = nx.relabel_nodes(DAG, mapping, copy=True)
    # get template source
    source = inspect.getsource(DAGModel.model).strip()
    tree = parse(source)
    # figure out which plates to create and in what order
    plate_graph = make_plate_graph(DAG)
    new_plates = list(nx.topological_sort(plate_graph))
    # create plates
    for plate_config in new_plates:
        prev_plate = plate_config[0]
        AddToFunctionBody(construct_plate(DAG, prev_plate), pos=-1).visit(tree)
        while plate_config[1:]:
            plate_config = plate_config[1:]
            plate = plate_config[0]
            AddToPlate(prev_plate, construct_plate(DAG, prev_plate+plate)).visit(tree)
            prev_plate = prev_plate+plate
    # create all nodes in DAG
    for node in nx.topological_sort(DAG):
        # if it's a root node, add it top level in the function
        if DAG.in_degree[node] == 0:
            # edit model tree
            if DAG.nodes[node]['type'] == 'param':
                AddToFunctionBody(construct_param(DAG, node), pos=0).visit(tree)
                AddToFunctionBody(construct_initalization(DAG, node), pos=0).visit(tree)
            elif DAG.nodes[node]['type'] == 'function':
                AddToFunctionBody(construct_function(DAG, node), pos=0).visit(tree)
            elif DAG.nodes[node]['type'] == 'const':
                AddToFunctionBody(construct_constant(DAG, node), pos=0).visit(tree)
        # if not, check for plating, and maintain dependency order unplated nodes
        else:
            if 'distribution' in DAG.nodes[node]:
                constructor = construct_sample(DAG, node)
            elif 'function' in DAG.nodes[node]:
                constructor = construct_function(DAG, node)
                
            if 'plates' in DAG.nodes[node]:
                plate = ''.join(DAG.nodes[node]['plates'])
                AddToPlate(plate, constructor).visit(tree)
            else:
                # WARNING: This is not general enough, will fail sooner or later
                parents = [edge[0] for edge in DAG.in_edges(node)]
                # find plate containing first parent of node
                if 'plates' in DAG.nodes[parents[0]]:
                    parent_plate = DAG.nodes[parents[0]]['plates']
                    # find index of plate containing parent of node
                    pos = [i for i,elem in enumerate(tree.body[0].body) if (type(elem) == ast.With and elem.items[0].context_expr.args[0].values[0].s == parent_plate[0]+'_')]
                    pos = pos[0] + 1
                # if parent is not plated, construct node after parent
                else:
                    pos = [i for i,elem in enumerate(tree.body[0].body) if (type(elem) == ast.Assign and elem.targets[0].id == parents[0])]
                    pos = pos[0] + 1
                AddToFunctionBody(constructor, pos=pos).visit(tree)
    # add _id to model
    AddToFunctionBody(Assign(targets=[Name(id='_id')], value=Attribute(value=Name(id='self'), attr='_id')), pos=0).visit(tree)
    # add shape assignment
    get_data_shape = Assign(targets=[Tuple(elts=[Name(id='N'), Name(id='D')])], value=Attribute(value=Name(id='X'), attr='shape'))
    AddToFunctionBody(get_data_shape).visit(tree)
    # add dim definitions
    for dim in dims:
        AddToFunctionBody(Assign(targets=[Name(id=dim)], value=Attribute(value=Name(id='self'), attr=dim))).visit(tree)
    return tree

def generate_get_param_shapes_and_support_and_init(DAG):
    param_shape_keys = []
    param_shape_vals = []
    shape_dims = []
    for node in nx.topological_sort(DAG):
        if DAG.nodes[node]['type'] == 'param':
            if 'constraint' in DAG.nodes[node]:
                constraint = DAG.nodes[node]['constraint']
            else:
                constraint = 'real'
                
            prior_param = '_'.join(node.split('_')[:-1] + ['prior'] + node.split('_')[-1:])
            
            param_shape_keys.append(JoinedStr(values=[Str(s=node+'_init_'),
                                FormattedValue(value=Name(id='_id'), conversion=-1, format_spec=None)]))
            param_shape_keys.append(JoinedStr(values=[Str(s=prior_param+'_init_'),
                                FormattedValue(value=Name(id='_id'), conversion=-1, format_spec=None)]))

            shape_elts = [Attribute(value=Name(id='self'), attr=dim) for dim in DAG.nodes[node]['shape']]
            param_shape_vals.append(Tuple(elts=[Tuple(elts=shape_elts),
                            Attribute(value=Name(id='constraints'), attr=constraint)]))
            param_shape_vals.append(Tuple(elts=[Tuple(elts=shape_elts),
                            Attribute(value=Name(id='constraints'), attr=constraint)]))
            if DAG.nodes[node]['shape'] not in shape_dims:
                shape_dims.append(DAG.nodes[node]['shape'])

    # construct get_param_shapes_and_support()
    get_param_shape_source = inspect.getsource(DAGModel.get_param_shapes_and_support).strip()
    get_param_shape_tree = parse(get_param_shape_source)
    
    AddReturn(Dict(keys=param_shape_keys,values=param_shape_vals)).visit(get_param_shape_tree)
    
    # edit __init__()
    init_source = inspect.getsource(DAGModel.__init__).strip()
    init_tree = parse(init_source)
    dims = [dim for shape in shape_dims for dim in shape if dim not in 'ND']
    for dim in dims:
        AddArgsToFunctionDef(dim,pos=2).visit(init_tree)
        AddToFunctionBody(Assign(targets=[Attribute(value=Name(id='self'), attr=dim)], value=Name(id=dim))).visit(init_tree)
    return get_param_shape_tree, init_tree, dims

def generate_guide(DAG, dims):
    guide_DAG = DAG.copy()
    # remove X
    # remove nodes connected to X that aren't latent nodes
    # properly we should recursively remove nodes that lead to X that aren't latent nodes until we reach latent nodes,
    # but in this case we just have one layer of deterministic nodes
    # so we'll worry about the general case later
    nodes_to_remove = ['X']
    in_nodes = [edge[0] for edge in guide_DAG.in_edges('X') if guide_DAG.nodes[edge[0]]['type'] != 'latent']
    nodes_to_remove.extend(in_nodes)
    guide_DAG.remove_nodes_from(nodes_to_remove)
    tree = generate_model(guide_DAG, dims)
    ChangeFunctionName('guide').visit(tree)
    CutFromFunctionBody(head=False).visit(tree)
    AddReturn(tuple([node for node in guide_DAG.nodes if guide_DAG.nodes[node]['type'] == 'latent'])).visit(tree)
    return tree

def generate_Model_class(DAG):
    get_param_shape_tree, init_tree, dims = generate_get_param_shapes_and_support_and_init(DAG)
    tree = generate_model(DAG, dims, root_node_suffix = 'prior')
    guide_tree = generate_guide(DAG, dims)

    # write source code
    fix_missing_locations(tree)
    model_source = astor.to_source(tree)
    
    fix_missing_locations(get_param_shape_tree)
    get_param_shape_source = astor.to_source(get_param_shape_tree)

    fix_missing_locations(init_tree)
    init_source = astor.to_source(init_tree)

    fix_missing_locations(guide_tree)
    guide_source = astor.to_source(guide_tree)

    class_source = inspect.getsource(DAGModel).strip()
    # insert source code
    class_source = insert_function_into_class(class_source, init_source, 'def __init__', 'super(DAGModel, self).__init__(X, batch_size, _id)')    
    class_source = insert_function_into_class(class_source, get_param_shape_source, 'def get_param_shapes_and_support', '_id = self._id')    
    class_source = insert_function_into_class(class_source, model_source, 'def model', 'return X')    
    class_source = insert_function_into_class(class_source, guide_source, 'def guide', 'raise NotImplementedError')    

    #class_source = parse(class_source)
    #fix_missing_locations(class_source)
    #class_source = astor.to_source(class_source)
    #print(class_source)
    # write to file
    with open("model.py", "w") as output:
        output.write('from models_and_guides import *\n\n')
        output.writelines(class_source)

