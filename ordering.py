import numpy as np
import time

from parameters import *
from helper_class import *
from scheduler_class import *

def order(G,M,O,T):
    #instantiate the nsoc
    platform = nsoc(n_cpus,n_npus)

    #operations
    n_ops = len(M)

    #start and end times
    start_times     = [-1 for i in range(n_ops)]
    end_times       = [-1 for i in range(n_ops)]
    computing_cores = [None for i in range(n_ops)]
    computing_ids   = [-1 for i in range(n_ops)]

    #dependancy matrix
    D = np.zeros((n_ops,n_ops))
    for i in range(len(G)):
        D[G[i,1],G[i,0]] = 1

    #define a few data structures
    active_ops          = []
    active_cores        = []
    active_ids          = []
    active_ops_status   = []
    active_ops_end      = []

    ready_ops   = []

    tick        = 0

    remaining_ops       = len([i for i, x in enumerate(end_times) if x<0])

    #start the show
    considered_ops      = active_ops + ready_ops
    D_sum               = list(np.sum(D,1))
    c                   = [i for i, x in enumerate(D_sum) if x==0]
    for i in c:
        if i not in considered_ops:
            ready_ops.append(i)
    while(remaining_ops > 0):    #the main while loop
        ########################################################
        #check if any of the running operations can be completed
        #end some schedule
        ########################################################
        completed_ops_idx = [i for i, x in enumerate(active_ops_end) if tick>=x]
        if len(completed_ops_idx) > 0:
            #complete individual operation
            for i in completed_ops_idx:
                opi         = active_ops[i]
                corei       = active_cores[i]
                core_idxi   = active_ids[i]
                active_ops_status[i] = 1
                op_end      = active_ops_end[i]
                opx         = O[opi]
                #deschedule the operation
                if corei == 'cpu':
                    platform.unlock_cpu(core_idxi,opx)
                elif corei == 'npu':
                    platform.unlock_npu(core_idxi,opx)
                #mark the end times
                end_times[opi] = op_end
                #update the dependency matrix
                D[:,opi] = 0
            
            #remove the completed operations from the active lists
            remove_idx          = [i for i, x in enumerate(active_ops_status) if x==1]  #indexes that are complete
            active_ops          = [i for j, i in enumerate(active_ops)          if j not in remove_idx]
            active_cores        = [i for j, i in enumerate(active_cores)        if j not in remove_idx]
            active_ids          = [i for j, i in enumerate(active_ids)          if j not in remove_idx]
            active_ops_status   = [i for j, i in enumerate(active_ops_status)   if j not in remove_idx]
            active_ops_end      = [i for j, i in enumerate(active_ops_end)      if j not in remove_idx]

            #check if any of the operations can be scheduled
            considered_ops          = active_ops + ready_ops + [i for i, x in enumerate(end_times) if x>=0]
            #if debug:
            #    print('considered_ops = ',considered_ops)
            D_sum                   = list(np.sum(D,1))
            c                       = [i for i, x in enumerate(D_sum) if x==0]
            for i in c:
                if i not in considered_ops:
                    ready_ops.append(i)
        ########################################################
        #check if any of the ready operations can be fired
        #start some schedule
        ########################################################
        if len(ready_ops) > 0:
            #if debug:
            #    print('Ready operations = ',ready_ops)
            delete_list = []
            for opi in ready_ops:
                #see if opi can be scheduled
                corei = M[opi]
                opx   = O[opi]
                #if debug:
                #    print('\t corei = ',corei, ' and opx = ',opx)
                if corei == 'cpu':
                    cpu_ids = platform.get_free_cpuid()
                    #if debug:
                    #    print('\t free cpu id = ',cpu_ids)
                    if len(cpu_ids) > 0:
                        cpuid = cpu_ids[0]
                        platform.lock_cpu(cpuid,opx)
                        active_ops.append(opi)
                        active_cores.append('cpu')
                        active_ids.append(cpuid)
                        active_ops_status.append(0)
                        active_ops_end.append(tick+T[opi])
                        #set the start time
                        start_times[opi]        = tick
                        computing_cores[opi]    = 'cpu'
                        computing_ids[opi]      = cpuid
                        #update the ready list
                        delete_list.append(opi)
                elif corei == 'npu':
                    npu_ids = platform.get_free_npuid(opx)
                    #if debug:
                    #    print('\t free npu id = ',npu_ids)
                    if len(npu_ids) > 0:
                        npuid = npu_ids[0]
                        platform.lock_npu(npuid,opx)
                        active_ops.append(opi)
                        active_cores.append('npu')
                        active_ids.append(npuid)
                        active_ops_status.append(0)
                        active_ops_end.append(tick+T[opi])
                        #set the start time
                        start_times[opi]        = tick
                        computing_cores[opi]    = 'npu'
                        computing_ids[opi]      = npuid
                        #update the ready list
                        delete_list.append(opi)
                elif corei == 'any':
                    npu_ids = platform.get_free_npuid(opx)
                    cpu_ids = platform.get_free_cpuid()
                    #if debug:
                    #    print('\t free npu id = ',npu_ids)
                    if len(npu_ids) > 0:
                        npuid = npu_ids[0]
                        platform.lock_npu(npuid,opx)
                        active_ops.append(opi)
                        active_cores.append('npu')
                        active_ids.append(npuid)
                        active_ops_status.append(0)
                        active_ops_end.append(tick+T[opi]['npu'])
                        #set the start time
                        start_times[opi]        = tick
                        computing_cores[opi]    = 'npu'
                        computing_ids[opi]      = npuid
                        #update the ready list
                        delete_list.append(opi)
                    elif len(cpu_ids) > 0:
                        cpuid = cpu_ids[0]
                        platform.lock_cpu(cpuid,opx)
                        active_ops.append(opi)
                        active_cores.append('cpu')
                        active_ids.append(cpuid)
                        active_ops_status.append(0)
                        active_ops_end.append(tick+T[opi]['cpu'])
                        #set the start time
                        start_times[opi]        = tick
                        computing_cores[opi]    = 'cpu'
                        computing_ids[opi]      = cpuid
                        #update the ready list
                        delete_list.append(opi)
            #update the ready list
            if len(delete_list) > 0:
                for el in delete_list:
                    ready_ops.remove(el)

    
        #book keeping
        tick += CLOCK
        remaining_ops       = len([i for i, x in enumerate(end_times) if x<0])
        
    sch = schedule(0)
    sch.fill_schedule(start_times,end_times,computing_cores,computing_ids,G,M,O,T)
    return sch
