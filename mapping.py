import numpy as np
from parameters import *
from helper_fns import *

def random_mapping(O,s):
    np.random.seed(s)
    n               = len(O)
    core_types      = ['cpu','npu']
    x               = np.random.randint(2, size=n)
    M               = {}
    for xi,i in zip(x,range(n)):
        if xi==1 and O[i] not in NPU_SUPPORTED_OPERATIONS:
            xi = 0
        M[i] = core_types[xi]
    return M

def npu_mapping(O,s):
    np.random.seed(s)
    n               = len(O)
    core_types      = ['cpu','npu']
    x               = np.ones(n).astype('int')
    M               = {}
    for xi,i in zip(x,range(n)):
        if xi==1 and O[i] not in NPU_SUPPORTED_OPERATIONS:
            xi = 0
        M[i] = core_types[xi]
    return M
