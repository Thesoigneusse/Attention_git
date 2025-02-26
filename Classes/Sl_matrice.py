import torch
import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git')

from Classes.Snt import Snt
from Classes.Matrice_deprecated_2 import Matrice
from typing import List

class Sl_matrice(Matrice):
    def __new__(cls, data = None, *args, **kwargs):
        instance = super().__new__(cls, data, *args, **kwargs) if data is not None else super().__new__(cls, 0, *args, **kwargs)
        return instance

    def __init__(self, matrice: torch.Tensor = None) -> None:
        pass
        # super().__init__(matrice)

    def __json__(self) -> str:
        return str(self.__dict__)

    def __mul__(self, other):
        assert isinstance(other, int), f"[DEBUG] operator __mul__ only supported on positive integers. Current type: {type(other)}"
        if isinstance(other, int) and other > 0:
            from copy import copy
            res = []
            for i in range(other):
                res.append(copy(self))
            return res


        # Écriture
    
    def contextualise_matrice(self, other: List[Matrice]) -> Matrice:
        """multiplie une matrice sentence-level avec 3 matrice word-level

        Args:
            other (List[Matrice]): liste des matrices word-level avec 3 matrices word-level

        Returns:
            Matrice: matrice contextualisée des 3 matrices words-level avec la matrice sentence-level
        Tests:
        >>> matrice = Sl_matrice(torch.Tensor([[1,2,3], [1,2,3], [1,2,3]]))
        >>> matrices = [Matrice([[1, 2], [3, 4], [5, 6]]), Matrice([[0.5, 1], [1.5, 2], [2.5, 3]]), Matrice([[2, 0.5], [1, 3], [6, 2]])]
        >>> print(matrice.contextualise_matrice(matrices))
        Matrice([[ 1.0000,  2.0000,  1.0000,  2.0000,  6.0000,  1.5000],
                 [ 3.0000,  4.0000,  3.0000,  4.0000,  3.0000,  9.0000],
                 [ 5.0000,  6.0000,  5.0000,  6.0000, 18.0000,  6.0000]])

        """
        assert isinstance(other, List), f"other must be an instance of Matrice. Current type: {type(other)}"
        assert self.size(0) == other[0].size(0), f"The number of rows of the Sl_matrice must be equal to the number of rows of the Matrice. Current shape: {self.size()}, {other[0].size()}"
        assert self.size(1) == len(other), f"The number of columns of the Sl_matrice must be equal to the number of columns of the Matrice. Current shape: {self.size()}, {other.size()}"
        # print(f"{other[0].size()} vs. {other[1].size()}")

        full_matrice = []
        for t in range(self.size(0)):
            # pour t le nombre de tokens dans la phrase courante (ligne de sl_matrice & de chaque matrice)
            temp_ctxs = []
            # print(f"[debug] self[t, k]: {self}")
            for k in range(self.size(1)):
                # pour k le nombre de contexte(colonne de self)
                temp_ctxs.append(torch.Tensor(other[k][t, ...] * self[t, k]))
            # print(f"{[ctx.size() for ctx in temp_ctxs]}")
            full_matrice.append(torch.cat(temp_ctxs))
        #print(full_matrice[0].size())

        full_matrice = torch.stack(full_matrice, dim=0)
        # for l in range(full_matrice.size(dim = 0)):
        #     print(f"[debug ligne l]{full_matrice[l, -10:]}")



        return Matrice(full_matrice)

    def ecriture_xslx(self, crt: Snt = None, ctx: Snt = Snt(identifiant= -1, tokens = ["k3", "k2", "k1"]), absolute_folder = None, filename = None, precision=2,  create_folder_path = False):
        super().ecriture_xslx(crt= crt, ctx=ctx, absolute_folder=absolute_folder, filename=filename, precision=precision, create_folder_path=create_folder_path)

    def test_(self, size = [10,10]):
        print(f"[DEBUG] Production d'une Sl_matrice de Test")
        super().test_(size)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    matrice = Sl_matrice(torch.Tensor([[1,2,3],
                                       [1,2,3],
                                       [1,2,3]]))
    matrices = [Matrice([[1, 2],
                         [3, 4],
                         [5, 6]]),
                Matrice([[0.5, 1],
                         [1.5, 2],
                         [2.5, 3]]),
                Matrice([[2, 0.5],
                         [1, 3],
                         [6, 2]])
               ]
    print(matrice)
    # print(matrices)
    # print(matrice.contextualise_matrice(matrices))

