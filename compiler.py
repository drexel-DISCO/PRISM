import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-model',           '--model',          default='resnet50')
parser.add_argument('-seed',            '--seed',           default='42')
parser.add_argument('-gpu',             '--gpu',            default='0')
parser.add_argument('-schd',            '--schd',           default='rnd')
parser.add_argument('-out',             '--out',            default='schedules')

args        = vars(parser.parse_args())
model_name  = args['model']
seed        = args['seed']
gpu_id      = args['gpu']
scheduler   = args['schd']
savedir     = args['out']

import pickle
import numpy as np
import time

import tensorflow as tf
import keras
import json
from keras.models import model_from_json
from keras import backend as K

from helper_fns import *
from scheduling import *
from scheduler_class import *

def scheduling_algorithm(scheduler,G,O,Tcpu,Tnpu,nbatch=BATCH_SIZE):
    if scheduler == 'rnd':
        sch = schedule_rnd(G,O,Tcpu,Tnpu,nbatch)
    elif scheduler == 'npufirst':
        sch = schedule_npu(G,O,Tcpu,Tnpu,nbatch)
    elif scheduler == 'prism':
        sch = schedule_prism(G,O,Tcpu,Tnpu,nbatch)
    else:
        print('Scheduler ', scheduler, ' is not currently supported')
        exit()
    if debug:
        s = sch[0]
        if 'npu' in s.computing_cores:
            print('Number of NPUs used = ',max([s.computing_ids[j] for j in [i for i, x in enumerate(s.computing_cores) if x=='npu']])+1)
        else:
            print('Number of NPUs used = 0')
        if 'cpu' in s.computing_cores:
            print('Number of CPUs used = ',max([s.computing_ids[j] for j in [i for i, x in enumerate(s.computing_cores) if x=='cpu']])+1)
        else:
            print('Number of CPUs used = 0')
    return sch

ts = time.time()
if GENERATE_FLATTENED_GRAPH:
    #read the input graph
    json_fname = 'jsons/'   +model_name + '.json'
    h5_fname   = 'h5s/'     +model_name + '.h5'
    json_file  = open(json_fname, 'r')
    model_json = json_file.read()
    json_file.close()
    model      = model_from_json(model_json)
    model.load_weights(h5_fname)
    #save the graph in pickle and as a flattened matrix
    model_graph = generate_pckled_graph(model)
    G           = flatten_graph(model_graph)
    #Get all operations
    O         = [n.get_type() for n in model_graph]
else:
    gname = 'cgraphs/' + model_name + '.pkl'
    with open(gname, "rb") as mf:
        G = pickle.load(mf)
    #read the operations
    oname = 'operations/' + model_name + '.pkl'
    with open(oname, "rb") as of:
        O = pickle.load(of)

#read the extime
ename = 'extime/' + model_name + '.pkl'
with open(ename, "rb") as ef:
    E = pickle.load(ef)
Tcpu  = [E[i]['cpu'] for i in range(len(O))]
Tnpu  = [E[i]['npu'] for i in range(len(O))]

initialization_time = time.time() - ts
ts = time.time()

#compute the schedule with the given BATCH SIZE and save it
print('[info] Scheduling ',model_name,' ...')
sch = scheduling_algorithm(scheduler,G,O,Tcpu,Tnpu)
ofname = savedir+'/'+model_name+'.pkl'
pickle.dump(sch,open(ofname,"wb"))
schedule_generation_time = time.time() - ts
print('[info] done')
print(model_name, ': execution time for ',BATCH_SIZE, ' batches = ',[s.get_completion_time() for s in sch], 'us')
print('Simulation time for graph compilation (wall clock time) = ',initialization_time, ' seconds')
print('Simulation time for schedule generation (wall clock time) = ',schedule_generation_time, ' seconds')

if scheduler == 'hco':
    exit()
ts = time.time()
#this part is to compute the average throughput
print('[info] Computing average throughput of ',model_name,' ...')
throughput = []
batch_sizes = [2,4,8,16,32,64,128] #explored batch sizes
for nbatch in batch_sizes:
    print('[info] Scheduling ',nbatch, ' batches')
    sch = scheduling_algorithm(scheduler,G,O,Tcpu,Tnpu,nbatch) #compute the schedule
    completion_time = [s.get_completion_time() for s in sch][0]     #compute the completion time
    thr = float(completion_time) / float(nbatch)
    throughput.append(thr)
print('[info] done')
throughput_computing_time = time.time() - ts

print(model_name, ': average model execution time per batch = ',np.mean(throughput), 'us')
print('Simulation time for throughput computation (wall clock time) = ',throughput_computing_time, ' seconds')
