from Classes.Matrice import Matrice
from Classes.Snt import Snt
from typing import List

class Src_matrice(Matrice):
    def __new__(cls, data = None, *args, **kwargs):
        instance = super().__new__(cls, data, *args, **kwargs) if data is not None else super().__new__(cls, 0, *args, **kwargs)
        return instance

    def __init__(self, data = None):
        pass

    def norm_tenseur(self, medium="minmax"):
        from Utils import action_norm_tensor as ant
        dict_action = {"minmax": ant.norm_by_min_max,
                        "max": ant.norm_by_max}
        for i in range(self.size(dim=0)):
            self[i] = dict_action[medium](self[i])
        return self

