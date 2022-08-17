import numpy as np
import random
import json
import numpy as np
from scipy import spatial
from functools import reduce
import pickle

from compiler_class import *
from parameters import *
from ordering import *
from scheduler_class import *

def mapping_aware_extime(M,Tc,Tn):
    T = []
    for i in range(len(M)):
        if M[i] == 'cpu':
            T.append(Tc[i])
        elif M[i] == 'npu':
            T.append(Tn[i])
        elif M[i] == 'any':
            d = {}
            d['cpu'] = Tc[i]
            d['npu'] = Tn[i]
            T.append(d)
    return T

def replicate_mapping(M,nbatch=BATCH_SIZE):
    M_keys    = list(M.keys())
    M_values  = list(M.values())

    new_keys  = []
    new_values= []

    for i in range(nbatch):
        new_keys    += [el + (i * len(M_keys)) for el in M_keys]
        new_values  += M_values
    
    new_M = {}
    for k,v in zip(new_keys,new_values):
        new_M[k] = v

    return new_M

def replicate_list(L,nbatch=BATCH_SIZE):
    new_L = []
    for i in range(nbatch):
        new_L += L
    return new_L

def replicate_graph(G,offset,nbatch=BATCH_SIZE):
    import copy
    n = G.shape[1]
    new_G = np.zeros((0,n)).astype('int')
    for i in range(nbatch):
        multiplier = copy.deepcopy(G)
        multiplier[:,0:2] += i*offset
        new_G = np.vstack((new_G,multiplier))
    return new_G

def is_mapping_valid(M,O):
    valid = True
    for key,value in M.items():
        op = O[key]
        if value == 'npu' and op not in NPU_SUPPORTED_OPERATIONS:
            valid = False
            #print(M)
            #print(key,value,op)
            #exit()
    return valid

def generate_pckled_graph(model):
    layers          = model.layers
    nLayers         = len(layers)
    config_layers   = model.get_config()["layers"]
    nConfigLayers   = len(model.get_config()["layers"])

    #get the layer names
    layer_names     = [layer['config']['name'] for layer in config_layers]
    #create the nodes
    model_graph     = [node(layer) for layer in layer_names]    #create node names
    #create node ids (for easier manipulation)
    idx = 0
    for n in model_graph:
        n.add_id(idx)
        idx += 1

    if nLayers != nConfigLayers:
        print('Model is sequential')
        #special processing for sequential model because the first layer in model.layer is not the InputLayer
        missing_layer_name = model.layers[0]._inbound_nodes[0].inbound_layers
        layers.insert(0, missing_layer_name)
    
    #print('Das Debug')
    #for each layer find the outgoing layers
    for layer in layers:
        layer_weights   = layer.get_weights()   #get layer weights
        layer_in_shape  = layer.input_shape     #get the shape of the input to the layer
        layer_out_shape = layer.output_shape    #get the shape of the output of the layer
        layer_type      = layer.__class__.__name__  #generic class of the layer
        #compute the state space
        #this is the input shape
        if isinstance(layer_in_shape, tuple):
            layer_in_shape = [layer_in_shape]   #change a tuple to a list for uniform processing
        state_space = []
        for list_el in layer_in_shape:
            sspace = 1
            for tuple_el in list_el:
                if tuple_el is not None:
                    sspace *= tuple_el
            state_space.append(sspace)
        #compute the output tokens
        #this is the output shape
        if isinstance(layer_out_shape, tuple):
            layer_out_shape = [layer_out_shape] #change a tuple to a list for uniform processing
        tokens = []
        for list_el in layer_out_shape:
            tk = 1
            for tuple_el in list_el:
                if tuple_el is not None:
                    tk *= tuple_el
            tokens.append(tk)
    
        outbound_layers = [on.outbound_layer.name for on in layer._outbound_nodes]  #get outgoing layers
        layer_index     = layer_names.index(layer.name)             #find the node index
    
        model_graph[layer_index].add_type(layer_type)               #set the generic layer type
        model_graph[layer_index].add_fanout_list(outbound_layers)   #set the outgoing layers as fanout of the node
        model_graph[layer_index].add_state_space(state_space)       #set the state space of the node
        model_graph[layer_index].add_tokens(tokens)                 #set the state space of the node
        for olayer in outbound_layers:                              #for each outhoing layer add the current layer as fanin
            olayer_index = layer_names.index(olayer)                #find the node index of the outbound layer
            model_graph[olayer_index].add_fanin(layer.name)         #add the current layer as the index
    
    #return the raw graph
    return model_graph

def flatten_graph(G):
    flattened_graph = []
    node_names      = [n.get_name() for n in G]

    for n in G:
        n_name  = n.get_name()
        n_idx   = node_names.index(n_name)
        n_out   = n.get_fanout()
        n_token = n.get_tokens()[0]
        for fo in n_out:
            o_idx = node_names.index(fo)
            flattened_graph += [n_idx,o_idx,n_token]
    flattened_graph = np.array(flattened_graph)
    flattened_graph = flattened_graph.reshape(-1,3)

    return flattened_graph
