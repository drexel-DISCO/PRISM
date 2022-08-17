import numpy as np

class schedule:
    def __init__(self,sch_id):
        self.idx                = sch_id
        self.start_times        = []
        self.end_times          = []
        self.computing_cores    = []
        self.computing_ids      = []
        self.graph              = None
        self.map                = None
        self.operation          = None
        self.extime             = None

    def fill_schedule(self,S,E,C,I,G,M,O,T):
        self.start_times        = S
        self.end_times          = E
        self.computing_cores    = C
        self.computing_ids      = I
        self.graph              = G
        self.map                = M
        self.operation          = O
        self.extime             = T

    def set_id(self,sch_id):
        self.idx = sch_id

    def get_completion_time(self):
        if len(self.end_times) == 0:
            return 0
        else:
            return max(self.end_times)


