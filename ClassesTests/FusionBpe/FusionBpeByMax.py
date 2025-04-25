from __future__ import annotations
from StrategyFusionBpe import StrategyFusionBpe
from icecream import ic
from typing import TYPE_CHECKING
from typing import List
import torch

if TYPE_CHECKING:
    import sys
    sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/ClassesTests')
    from Matrice import Matrice
    from BaseSentence import BaseSentence

class FusionBpeByMax(StrategyFusionBpe):

    def fuse_matrice(self, matrice: Matrice, list_row_index_bpe: List[List[int]], list_column_index_bpe: List[List[int]]) -> Matrice:
        """Fusionne la matrice en utilisant ne gardant que le maximum des poids des différents BPEs d'un mot.

        Args:
            matrice (Matrice): La matrice à fusionner.

        Returns:
            Matrice: La matrice fusionnée.
        """
        raise "BUG dans le cas d'un index comprennant l'index du dernier élément de la matrice"
        # Traitement des BPEs sur les lignes
        for groupe_index in list_row_index_bpe:
            print("before")
            ic(matrice)
            assert len(groupe_index) > 1, f"list_row_index_bpe must be list of at least 2 elements. Current element length: {len(groupe_index)}"
            matrice[groupe_index[0], ...] = torch.max(matrice[groupe_index, ...], dim = 0).values
            matrice = matrice.suppr_lignes_from_i1_to_i2(groupe_index[1], groupe_index[-1]+1)
            print("after")
            ic(matrice)
        print("****************")
        # Traitement des BPEs sur les colonnes
        for groupe_index in list_column_index_bpe:
            print("before")
            ic(matrice)
            assert len(groupe_index) > 1, f"list_column_index_bpe must be list of at least 2 element. Current element length: {len(groupe_index)}"
            ic(matrice[:, groupe_index])
            matrice[:, groupe_index[0]] = torch.max(matrice[:, groupe_index], dim = 1).values
            matrice = matrice.suppr_colonnes_from_c1_to_c2(groupe_index[1], groupe_index[-1]+1)
            print("after")
            ic(matrice)
        return matrice

def main():
    import sys
    sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/ClassesTests')
    from BaseSentence import BaseSentence
    from BaseWord import BaseWord
    from Matrice import Matrice
    """Main function to test the FusionBpeByMax class."""
    ic("main function of FusionBpeByMax")
    sentence = BaseSentence(identifiant=1, tokens=[BaseWord(identifiant= 0, token= "Hel@@"), 
                                                   BaseWord(identifiant= 1, token= "lo"),
                                                   BaseWord(identifiant=2, token= "world")])
    matrice = Matrice(matrice= torch.Tensor([[1,2,3,4,5,6],
                                            [7,8,9,10,11,12],
                                            [13,14,15,16,17,18],
                                            [19,20,21,22,23,24],
                                            [25,26,27,28,29,30],
                                            [31,32,33,34,35,36]]))
    bpe_mark = "@@"
    fusion_bpe = FusionBpeByMax()
    fusion_bpe.fuse_token(sentence, bpe_mark)
    matrice = fusion_bpe.fuse_matrice(matrice=matrice, list_row_index_bpe=[[0,1], [2, 3]], list_column_index_bpe=[[1, 2], [3, 4, 5]])
    ic(matrice)
    print(f"Fused Sentence: {sentence.tokens}")

if __name__ == '__main__':
    main()