
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List
import torch

import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git')

@dataclass
class Matrice():
    matrice: torch.Tensor = field(default_factory=lambda: torch.Tensor())

    def __getitem__(self, key: int) -> torch.Tensor:
        """Permet d'accéder aux éléments ou tranches de la matrice via l'opérateur [].

        Args:
            key (int | slice | tuple): Indice ou slice pour accéder aux éléments.

        Returns:
            torch.Tensor: Élément(s) de la matrice correspondant à l'indice ou au slice.
        """
        return self.matrice[key]
    
    def __setitem__(self, key: int, value: torch.Tensor) -> None:
        """Permet de modifier les éléments ou tranches de la matrice via l'opérateur [].

        Args:
            key (int | slice | tuple): Indice ou slice pour accéder aux éléments.
            value (torch.Tensor): Valeur à assigner à l'élément(s) de la matrice.
        """
        self.matrice[key] = value

    def suppr_ligne_i(self, index: int) -> torch.Tensor:
        """Supprime la ligne en position index de la matrice.

        Args:
            index (int): index de la ligne à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor de la ligne supprimée.
        Tests:
        >>> matrice = Matrice(matrice = torch.Tensor([[1,2,3], [4,5,6]]))
        >>> print(matrice.suppr_ligne_i(0))
        Matrice(matrice=tensor([[4., 5., 6.]]))
        """
        return Matrice(matrice=torch.cat([self.matrice[:index,...], self.matrice[index+1:, ...]]))

    def suppr_colonne_i(self, index: int) -> torch.Tensor:
        """Supprime la colonne en position index de la matrice.

        Args:
            index (int): index de la colonne à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor de la colonne supprimée.
        Tests:
        >>> matrice = Matrice(matrice = torch.Tensor([[1,2,3], [4,5,6]]))
        >>> print(matrice.suppr_colonne_i(0))
        Matrice(matrice=tensor([[2., 3.],
                [5., 6.]]))
        """
        return Matrice(matrice=torch.cat([self.matrice[:, :index], self.matrice[:, index+1:]], dim=1))

    def suppr_lignes_from_i1_to_i2(self, index1, index2) -> torch.Tensor:
        """Supprime les ligne de la position index1 à index2 de la matrice.

        Args:
            index1 (int): index (inclu) du début du bloc de lignes à supprimer.
            index2 (int): index (exclu) de fin du bloc de lignes à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor des lignes supprimées.
        Tests:
        >>> matrice = Matrice(matrice = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]]))
        >>> print(matrice.suppr_lignes_from_i1_to_i2(0, 1))
        Matrice(matrice=tensor([[4., 5., 6.],
                [7., 8., 9.]]))
        """
        return Matrice(matrice=torch.cat([self.matrice[:index1,...], self.matrice[index2:, ...]]))

    def suppr_colonnes_from_c1_to_c2(self, index1, index2) -> torch.Tensor:
        """Supprime les colonnes de la position index1 à index2 de la matrice.

        Args:
            index1 (int): index (inclu) du début du bloc de colonnes à supprimer.
            index2 (int): index (exclu) de fin du bloc de colonnes à supprimer.

        Returns:
            torch.Tensor: Torch.Tensor des colonnes supprimées.
        Tests:
        >>> matrice = Matrice(matrice = torch.Tensor([[1,2,3], [4,5,6], [7,8,9]]))
        >>> print(matrice.suppr_colonnes_from_c1_to_c2(0, 1))
        Matrice(matrice=tensor([[2., 3.],
                [5., 6.],
                [8., 9.]]))
        >>> print(matrice.suppr_colonnes_from_c1_to_c2(0, 2))
        Matrice(matrice=tensor([[3.],
                [6.],
                [9.]]))
        >>> print(matrice.suppr_colonnes_from_c1_to_c2(1, 3))
        Matrice(matrice=tensor([[1.],
                [4.],
                [7.]]))

        """
        # Vérification des indices
        if index1 >= index2:
            raise ValueError(f"index1 ({index1}) doit être strictement inférieur à index2 ({index2}).")
        if index1 < 0 or index2 > self.matrice.size(1):
            raise ValueError(f"Les indices doivent être dans l'intervalle [0, {self.matrice.size(1)}].")
        return Matrice(matrice=torch.cat([self.matrice[:, :index1], self.matrice[:, index2:]], dim=1))


    @staticmethod
    def load_matrice(nb_row, nb_col):
        """Charge une matrice de taille nb_row x nb_col.

        Args:
            nb_row (int): Nombre de lignes de la matrice.
            nb_col (int): Nombre de colonnes de la matrice.

        Returns:
            Matrice: Matrice de taille nb_row x nb_col.
        
        Tests:
        >>> print(Matrice.load_matrice(2, 3))
        Matrice(matrice=tensor([[0., 1., 2.],
                [3., 4., 5.]]))
        """
        return Matrice(matrice= torch.Tensor([[i*nb_col+j for j in range(nb_col)] for i in range(nb_row)]))

    # def merge_bpe(list_merge_bpe: List[int], strategy: StrategyFusionBpe) -> None:

def main():
    from icecream import ic
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
    m1 = Matrice(matrice=t1)
    ic(Matrice.load_matrice(nb_row=10, nb_col=10))

    

if __name__ == '__main__':
    import doctest; doctest.testmod()
    print(f"[DEBUG]Doctest passed for BaseMatrice")
    main()
    # import cProfile

    # cProfile.run(main())
