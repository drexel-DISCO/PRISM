import numpy as np

class node:
    def __init__(self,name):
        self.name   = name
        self.ltype  = None
        self.idx    = -1
        self.sspace = -1
        self.tokens = -1
        self.fanin  = []
        self.fanout = []

    #function definitions
    #add functions
    def add_id(self,idx):
        self.idx    = idx

    def add_type(self,layer_type):
        self.ltype = layer_type

    def add_state_space(self,sspace):
        self.sspace = sspace

    def add_tokens(self,tokens):
        self.tokens = tokens

    def add_fanin(self,fanin):
        self.fanin.append(fanin)

    def add_fanin_list(self,fanin):
        self.fanin += fanin
    
    def add_faout(self,faout):
        self.faout.append(faout)

    def add_fanout_list(self,fanout):
        self.fanout += fanout

    def add_suffix(self,suffix):
        self.name   = self.name + suffix
        self.fanin  = [fanin + suffix for fanin in self.fanin]
        self.fanout = [fanout + suffix for fanout in self.fanout]
    
    #get functions
    def get_name(self):
        return self.name

    def get_id(self):
        return self.idx

    def get_type(self):
        return self.ltype

    def get_state_space(self):
        return self.sspace

    def get_tokens(self):
        return self.tokens

    def get_fanin(self):
        return self.fanin

    def get_fanout(self):
        return self.fanout
