
from abc import ABC, abstractmethod

class StrategyFusionBpe():

    @abstractmethod
    def fuse(self, tokens):
        pass