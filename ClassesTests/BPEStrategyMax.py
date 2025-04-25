from BaseBpeStrategy import BaseBpeStrategy
from ClassesTests.AttentionMatrice import BaseMatrice
from typing import List 
import torch


class BPEStrategyMax(BaseBpeStrategy):
    def merge(self, matrice: BaseMatrice, bpe_on_row: List[List[int]], bpe_on_col: List[List[int]]) -> BaseMatrice:
        """Fusionne les BPEs de la matrice.

        Args:
            matrice (BaseMatrice): Matrice à fusionner.
            bpe_on_row (List[List[int]]): Groupe d'indices des BPEs sur les lignes.
            bpe_on_col (List[List[int]]): Groupe d'indices des BPEs sur les colonnes.

        Returns:
            BaseMatrice: Matrice fusionnée.
        """
        
        for groupe_index in bpe_on_row:
            matrice.matrice[groupe_index[0]] = matrice.matrice[groupe_index].max(dim=0)[0]
            matrice.suppr_lignes_from_i1_to_i2(groupe_index[1], groupe_index[-1])
        for groupe_index in bpe_on_col:
            
