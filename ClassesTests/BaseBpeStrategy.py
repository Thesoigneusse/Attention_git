import torch
from typing import List

class BaseBpeStrategy():

    def merge(self, matrice: torch.Tensor, bpe_on_row: List[List[int]] , bpe_on_col: List[List[int]]) -> torch.Tensor:
        """Fusionne les BPE de la matrice.

        Args:
            matrice (torch.Tensor): Matrice à fusionner.
            bpe_on_row (bool): True si le BPE est sur les lignes, False sinon.
            bpe_on_col (bool): True si le BPE est sur les colonnes, False sinon.

        Returns:
            torch.Tensor: Matrice fusionnée.
        """
        raise NotImplementedError