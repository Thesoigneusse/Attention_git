from dataclasses import dataclass, field
from typing import List
import torch

import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git')
from Classes.Sentence import Sentence

@dataclass
class Matrice():
    current_sentence:Sentence = field(default_factory=Sentence)
    context_sentence:Sentence = field(default_factory=Sentence)
    matrice: torch.Tensor = field(default_factory=lambda: torch.Tensor())

    def suppr_ligne_i(self, index: int) -> torch.Tensor:
        """Supprime la ligne en position index de la matrice.

        Args:
            index (int): index de la ligne à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor de la ligne supprimée.
        Tests:
        # >>> matrice = Matrice(current_sentencetorch.Tensor([[1,2,3], [4,5,6]]))
        # >>> print(matrice.suppr_ligne_i(0))
        tensor([[4., 5., 6.]])
        """
        return torch.cat([self.matrice[:index,...], self.matrice[index+1:, ...]])

    def suppr_lignes_from_i1_to_i2(self, index1, index2):
        """Supprime les ligne de la position index1 à index2 de la matrice.

        Args:
            index1 (int): index (inclu) du début du bloc de lignes à supprimer.
            index2 (int): index (exclu) de fin du bloc de lignes à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor des lignes supprimées.
        Tests:
        # >>> matrice = Matrice(torch.Tensor([[1,2,3], [4,5,6], [7,8,9]]))
        # >>> print(matrice.suppr_lignes_from_i1_to_i2(0, 1))
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
        """
        return Matrice(current_sentence=Sentence(identifiant=1, tokens= [str(i) for i in range(nb_row)]),
                       context_sentence=Sentence(identifiant=0, tokens= [str(j) for j in range(nb_col)]),
                       matrice= torch.Tensor([[i*nb_col+j for j in range(nb_col)] for i in range(nb_row)]))


def main():
    from icecream import ic
    s1 = Sentence(identifiant= 3, 
                  tokens= ["<pad>", "<pad>", "Ce@@", "ci", "est", "un", "te@@", "st", ".", "<eos>"])
    s2 = Sentence(identifiant= 2,
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
    m1 = Matrice(current_sentence=s1, 
                context_sentence=s2, 
                matrice=t1)
    ic(Matrice.load_matrice(nb_row=10, nb_col=10))

    

if __name__ == '__main__':
    import doctest; doctest.testmod()
    print(f"[DEBUG]Doctest passed for Matrice")
    main()
    # import cProfile

    # cProfile.run(main())
