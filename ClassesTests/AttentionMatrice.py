
from dataclasses import dataclass, field
from typing import List
import torch

import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git')
from BaseSentence import BaseSentence

@dataclass
class BaseMatrice():
    current_sentence:BaseSentence = field(default_factory=BaseSentence)
    context_sentence:BaseSentence = field(default_factory=BaseSentence)
    matrice: torch.Tensor = field(default_factory=lambda: torch.Tensor())

    def suppr_ligne_i(self, index: int) -> torch.Tensor:
        """Supprime la ligne en position index de la matrice.

        Args:
            index (int): index de la ligne à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor de la ligne supprimée.
        Tests:
        >>> matrice = BaseMatrice(current_sentence = BaseSentence.load_BaseSentence(2), context_sentence = BaseSentence.load_BaseSentence(3), matrice = torch.Tensor([[1,2,3], [4,5,6]]))
        >>> print(matrice.suppr_ligne_i(0))
        tensor([[4., 5., 6.]])
        """
        return torch.cat([self.matrice[:index,...], self.matrice[index+1:, ...]])

    def suppr_colonne_i(self, index: int) -> torch.Tensor:
        """Supprime la colonne en position index de la matrice.

        Args:
            index (int): index de la colonne à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor de la colonne supprimée.
        Tests:
        >>> matrice = BaseMatrice(current_sentence= BaseSentence.load_BaseSentence(3), context_sentence = BaseSentence.load_BaseSentence(3), matrice = torch.Tensor([[1,2,3], [4,5,6]]))
        >>> print(matrice.suppr_colonne_i(0))
        tensor([[2., 3.],
                [5., 6.]])
        """
        return torch.cat([self.matrice[:, :index], self.matrice[:, index+1:]], dim=1)

    def suppr_lignes_from_i1_to_i2(self, index1, index2):
        """Supprime les ligne de la position index1 à index2 de la matrice.

        Args:
            index1 (int): index (inclu) du début du bloc de lignes à supprimer.
            index2 (int): index (exclu) de fin du bloc de lignes à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor des lignes supprimées.
        Tests:
        >>> matrice = BaseMatrice(current_sentence= BaseSentence.load_BaseSentence(3), context_sentence = BaseSentence.load_BaseSentence(3), matrice = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]]))
        >>> print(matrice.suppr_lignes_from_i1_to_i2(0, 1))
        tensor([[4., 5., 6.],
                [7., 8., 9.]])
        """
        return torch.cat([self.matrice[:index1,...], self.matrice[index2:, ...]])

    def suppr_colonnes_from_c1_to_c2(self, index1, index2):
        """Supprime les colonnes de la position index1 à index2 de la matrice.

        Args:
            index1 (int): index (inclu) du début du bloc de colonnes à supprimer.
            index2 (int): index (exclu) de fin du bloc de colonnes à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor des colonnes supprimées.
        Tests:
        >>> matrice = BaseMatrice(current_sentence= BaseSentence.load_BaseSentence(3), context_sentence = BaseSentence.load_BaseSentence(3), matrice = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]]))
        >>> print(matrice.suppr_lignes_from_i1_to_i2(0, 1))
        tensor([[4., 5., 6.],
                [7., 8., 9.]])
        """
        return torch.cat([self.matrice[:index1,...], self.matrice[index2:, ...]])


    @staticmethod
    def load_matrice(nb_row, nb_col):
        """Charge une matrice de taille nb_row x nb_col.

        Args:
            nb_row (int): Nombre de lignes de la matrice.
            nb_col (int): Nombre de colonnes de la matrice.

        Returns:
            Matrice: Matrice de taille nb_row x nb_col.
        
        Tests:
        >>> print(BaseMatrice.load_matrice(2, 3))
        BaseMatrice(current_sentence=BaseSentence(identifiant=1, tokens=['0', '1']), context_sentence=BaseSentence(identifiant=0, tokens=['0', '1', '2']), matrice=tensor([[0., 1., 2.],
                [3., 4., 5.]]))
        """
        return BaseMatrice(current_sentence=BaseSentence(identifiant=1, tokens= [str(i) for i in range(nb_row)]),
                           context_sentence=BaseSentence(identifiant=0, tokens= [str(j) for j in range(nb_col)]),
                           matrice= torch.Tensor([[i*nb_col+j for j in range(nb_col)] for i in range(nb_row)]))

    # def merge_bpe(strategy)

def main():
    from icecream import ic
    s1 = BaseSentence(identifiant= 3, 
                  tokens= ["<pad>", "<pad>", "Ce@@", "ci", "est", "un", "te@@", "st", ".", "<eos>"])
    s2 = BaseSentence(identifiant= 2,
                  tokens= ["<pad>", "<pad>", "Ce@@", "ci", "est", "un", "te@@", "st", ".", "<eos>"])
    t1 = torch.tensor([[ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                       [10,11,12,13,14,15,16,17,18,19],
                       [20,21,22,23,24,25,26,27,28,29],
                       [30,31,32,33,34,35,36,37,38,39],
                       [40,41,42,43,44,45,46,47,48,49],
                       [50,51,52,53,54,55,56,57,58,59],
                       [60,61,62,63,64,65,66,67,68,69],
                       [70,71,72,73,74,75,76,77,78,79],
                       [80,81,82,83,84,85,86,87,88,89],
                       [90,91,92,93,94,95,96,97,98,99]])
    m1 = BaseMatrice(current_sentence=s1, 
                context_sentence=s2, 
                matrice=t1)
    ic(BaseMatrice.load_matrice(nb_row=10, nb_col=10))

    

if __name__ == '__main__':
    import doctest; doctest.testmod()
    print(f"[DEBUG]Doctest passed for BaseMatrice")
    main()
    # import cProfile

    # cProfile.run(main())
