from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sys
    sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/ClassesTests')
    from Matrice import Matrice
    from BaseSentence import BaseSentence
from icecream import ic

class StrategyFusionBpe(ABC):
    
    def fuse_token(self, sentence: "BaseSentence", BPE_mark) -> "BaseSentence":
        """Fusionne les tokens de la phrase.

        Args:
            sentence (BaseSentence): La phrase à fusionner.
            BPE_mark (str): Le marqueur BPE à utiliser pour la fusion.

        Returns:
            BaseSentence: La phrase fusionnée.
        """
        for i in range(len(sentence.tokens) -1 , -1, -1):
            if sentence.tokens[i].endswith(BPE_mark):
                sentence.tokens[i] = sentence.tokens[i][:-len(BPE_mark)] + sentence.tokens[i+1]
                sentence.tokens.remove(sentence.tokens[i+1])

    @abstractmethod
    def fuse_matrice(self, matrice: "Matrice") -> "Matrice":
        pass

def main():
    ic("main function of StrategyFusionBpe")

if __name__ == '__main__':
    main()