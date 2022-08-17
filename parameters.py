debug = False
GENERATE_FLATTENED_GRAPH = True
PIPELINING = False

CLOCK = 100  #clock period in micro seconds

SPECK_SUPPORTED_OPERATIONS  = []
MUBRAIN_SUPPORTED_OPERATIONS = []
GRAI_SUPPORTED_OPERATIONS = ['InputLayer', 'Conv2D', 'MaxPooling2D', 'Flatten', 'Dense', 'BatchNormalization', 'ZeroPadding2D', 'Dropout', 'Activation', 'AveragePooling2D', 'Concatenate', 'GlobalAveragePooling2D', 'ReLU', 'DepthwiseConv2D', 'Add']

NPU_SUPPORTED_OPERATIONS = GRAI_SUPPORTED_OPERATIONS
starting_seed = 42
n_cpus = 2
n_npus = 4
BATCH_SIZE = 16
