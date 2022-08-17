import numpy as np
import time
import copy

from parameters import *
from mapping import *
from ordering import *
from helper_fns import *
from scheduler_class import *

def schedule_rnd(G,O,Tc,Tn,nbatch=BATCH_SIZE):
    all_schedules = []

    for s in [starting_seed]:
        M   = random_mapping(O,s)
        T   = mapping_aware_extime(M,Tc,Tn)
        sch = unit_schedule(G,M,O,T,s,nbatch)
        sch.set_id(s)

        all_schedules.append(sch)

    return all_schedules

def schedule_npu(G,O,Tc,Tn,nbatch=BATCH_SIZE):
    all_schedules = []

    for s in [starting_seed]:
        M   = npu_mapping(O,s)
        T   = mapping_aware_extime(M,Tc,Tn)
        sch = unit_schedule(G,M,O,T,s,nbatch)
        sch.set_id(s)

        all_schedules.append(sch)

    return all_schedules

def schedule_prism(G,O,Tc,Tn,nbatch=BATCH_SIZE):
    sch = schedule_rnd(G,O,Tc,Tn,nbatch)[0]
    
    Mstart  = {i:sch.computing_cores[i] for i in range(len(O))}
    Pstart  = sch.get_completion_time()

    print('[hco] Starting the Hill Climbing Optimization')
    hco_pass = True
    pass_cnt = 0
    while hco_pass:
        print('[hco] Pass ',pass_cnt,' performance = ',Pstart)
        pass_cnt += 1
        hco_pass = False
        for i in range(len(O)):
            Miter = copy.deepcopy(Mstart)
            if Miter[i] == 'cpu':
                Miter[i] = 'npu'
            elif Miter[i] == 'npu':
                Miter[i] = 'cpu'

            if is_mapping_valid(Miter,O):
                T   = mapping_aware_extime(Miter,Tc,Tn)
                sch = unit_schedule(G,Miter,O,T,starting_seed,nbatch)
                if sch.get_completion_time() < Pstart:
                    hco_pass = True
                    Mstart = copy.deepcopy(Miter)
                    Pstart = sch.get_completion_time()
                    print('[hco] New performance = ',Pstart)

    print('[hco] Endiing the Hill Climbing Optimization')
    T   = mapping_aware_extime(Mstart,Tc,Tn)
    sch = unit_schedule(G,Mstart,O,T,starting_seed,nbatch)

    return [sch]


def unit_schedule(G,M,O,T,s,nbatch=BATCH_SIZE):
   if nbatch > 1:
       G = replicate_graph(G,len(M),nbatch)
       M = replicate_mapping(M,nbatch)
       O = replicate_list(O,nbatch)
       T = replicate_list(T,nbatch)
   sch = order(G,M,O,T)
   sch.set_id(s)
   return sch
