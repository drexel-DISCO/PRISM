import numpy as np
from parameters import *

class core:
    def __init__(self,core_type,core_id):
        self.type       = core_type
        self.idx        = core_id
        self.status     = 0
        self.operation  = None

    def set_operation(self,operation):
        self.operation  = operation

    def lock_core(self):
        if self.status == 0:
            self.status = 1
            return True
        else:
            return False

    def unlock_core(self):
        if self.status == 1:
            self.status     = 0
            return True
        else:
            return False

    def get_status(self):
        return self.status

    def get_id(self):
        return self.idx

    def get_operation(self):
        return self.operation

class nsoc:
    def __init__(self,ncpus,nnpus):
        #set the cpus
        self.cpus = [core('cpu',i) for i in range(ncpus)]
        for icpu in self.cpus:
            icpu.set_operation('any')
        #set the npus
        self.npus = [core('npu',i) for i in range(nnpus)]
        for inpu in self.npus:
            inpu.set_operation('any')

    def lock_cpu(self,cpuid,op):
        lock_return = self.cpus[cpuid].lock_core()
        if lock_return:
            #print(op,' is scheduled on CPU ',cpuid)
            self.cpus[cpuid].set_operation(op)
            return True
        else:
            print('Error! CPU ',cpuid, ' is busy')
            exit()

    def lock_npu(self,npuid,op):
        lock_return = self.npus[npuid].lock_core()
        if lock_return:
            #print(op,' is scheduled on NPU ',npuid)
            self.npus[npuid].set_operation(op)
            return True
        else:
            print('Error! NPU ',npuid, ' is busy')
            exit()

    def unlock_cpu(self,cpuid,op):
        cpu_op          = self.cpus[cpuid].get_operation()
        cpu_status      = self.cpus[cpuid].get_status()
        if cpu_status == 1 and op == cpu_op:
            unlock_return   = self.cpus[cpuid].unlock_core()
            self.cpus[cpuid].set_operation('any')
            #print(op,' is descheduled on CPU ',cpuid)
        else:
            if cpu_status == 0 and op == cpu_op:
                print('Error! CPU ',cpuid, ' is idle')
                exit()
            elif op != cpu_op:
                print('Error! ',op,' is not mapped to CPU ',cpuid)
                exit()

    def unlock_npu(self,npuid,op):
        npu_op          = self.npus[npuid].get_operation()
        npu_status      = self.npus[npuid].get_status()
        if npu_status == 1 and op == npu_op:
            unlock_return   = self.npus[npuid].unlock_core()
            self.npus[npuid].set_operation('any')
            #print(op,' is descheduled on NPU ',npuid)
        else:
            if npu_status == 0 and op == npu_op:
                print('Error! NPU ',npuid, ' is idle')
                exit()
            elif op != npu_op:
                print('Error! ',op,' is not mapped to NPU ',npuid)
                exit()

    def get_cpu_status(self):
        return [cpui.get_status() for cpui in self.cpus]

    def get_npu_status(self):
        return [npui.get_status() for npui in self.npus]

    def get_free_cpuid(self):
        return [i for i, cpui in enumerate(self.cpus) if cpui.get_status()==0]

    def get_free_npuid(self,op):
        if op in NPU_SUPPORTED_OPERATIONS:
            return [i for i, npui in enumerate(self.npus) if npui.get_status()==0]
        else:
            return []

