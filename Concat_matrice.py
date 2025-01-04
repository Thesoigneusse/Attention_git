from Matrice import Matrice
from CA_matrice import CA_matrice
from Snt import Snt
from typing import List
# N : taille de la phrase courante
# k : nombre de phrase de contexte
# M_k : Taille de la phrase de contexte k
# M : Taille de la fusion des phrases de contexte
# nb_heads : nombre de tête d'attention
# L : nombre de layers


class Concat_matrice(CA_matrice):
    def __init__(self, crt: Snt, ctxs: List[Snt], layers: List[List[Matrice]]):
        super().__init__(crt=crt, ctxs=ctxs)
        
        # Variable correspondant aux layers des mécanismes d'attention 
        # Dimension : L x nb_heads x N x M

        self.layers = layers

    @property
    def layers(self) -> List[List[Matrice]]:
        return self._layers
    @layers.setter
    def layers(self, value: List[List[Matrice]]) -> None:
        assert isinstance(value, list), f"layers must be a list. Current Value: {type(value)}"
        assert all(isinstance(layer, List[Matrice]) for layer in value), f"layers must be a list of list. Current Value: {[type(layer) for layer in value]}"
        assert all(all(isinstance(matrice, Matrice) for matrice in layer) for layer in value), f"layers must be a list of list of Matrice. Current Value: {[type(layer[0]) for layer in value]}"
        self._layers = value

    





if __name__ == '__main__':
    import doctest
    import torch
    import Utils_data
    doctest.testmod()
    torch.set_printoptions(precision=2)
    print(f"[DEBUG] Doctest clear")
    _DEBUG_START = True
    _DEBUG_SUPPR_PAD = True
    _DEBUG_NORM_TENSOR = True
    _DEBUG_FUSION_BPE= False
    _PRECISION = 3
    _OUTPUT_PATH=f"/home/getalp/lopezfab/Documents"
    id = 1850
    r_path=f"/home/getalp/lopezfab/lig/temp/temp/test_attn/{id}.json"
    data=Utils_data.lecture_data(r_path)
    crt, ctxs, ctxs_heads, sl_heads = Utils_data.lecture_concat_objet(data)

    
