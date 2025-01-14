from Classes.Matrice import Matrice
from Classes.CA_matrice import CA_matrice
from Classes.Snt import Snt
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
    def layers(self, layers: List[List[Matrice]]) -> None:
        assert isinstance(layers, list), f"layers must be a list. Current Value: {type(layers)}"
        assert all(isinstance(layer, List) for layer in layers), f"layers must be a list of list. Current Value: {[type(layer) for layer in layers]}"
        assert all(all(isinstance(head, Matrice) for head in layer) for layer in layers), f"layers must be a list of list of Matrice. Current Value: {[type(layer[0]) for layer in layers]}"
        self._layers = layers

    def fusion_bpe(self, action: str = "max") -> None:
        """Fusionne les BPEs pour l'ensemble des matrices des chaque layers et de chaque tête d'attention

        Args:
            action (str, optional): action à appliquer sur la fusion des BPEs. Defaults to "max".
        """
        # Fusion des bpes pour les phrases courante et de contextes
        full_snt = self.get_full_snt()
        snt_list_fusion_bpe = self.get_full_snt().fusion_bpe()

        self.crt.fusion_bpe()
        for k in range(len(self.ctxs)):
            self.ctxs[k].fusion_bpe()

        # Fusion des bpes pour les matrices d'attention de chaque layer et de chaque head
        print(f"[DEBUG] full_snt: {full_snt}")
        print(f"[DEBUG] snt_list_fusion_bpe: {snt_list_fusion_bpe}")
        assert len(full_snt) == self.layers[0][0].size(dim=0), f'taille non compatible. len(full_snt) vs. len(self.layers): {len(full_snt)} vs. {self.layers[0][0].size(dim = 0)}'
        assert len(full_snt) == self.layers[0][0].size(dim=1), f'taille non compatible. len(full_snt) vs. len(self.layers): {len(full_snt)} vs. {self.layers[0][0].size(dim = 1)}'

        for i_layer in range(len(self.layers)):
            for h_head in range(len(self.layers[i_layer])):
                self.layers[i_layer][h_head].fusion_bpe(row_list_fusion_bpe=snt_list_fusion_bpe, col_list_fusion_bpe=snt_list_fusion_bpe)
        
    def norm_tensor(self, medium = "minmax") -> None:
        """Applique une normalisation sur les matrices des chaque layers et de chaque tête d'attention

        Args:
            medium (str, optional): médium de normalisation. Defaults to "minmax".
        """
        # Normalisation pour les matrices d'attention de chaque layer et de chaque head
        for i_layer in range(len(self.layers)):
            for h_head in range(len(self.layers[i_layer])):
                self.layers[i_layer][h_head].norm_tensor(medium=medium)

    def get_full_snt(self):
        full_snt = Snt(identifiant=-1, tokens=[])
        for ctx in self.ctxs:
            full_snt += ctx
        full_snt += self.crt
        return full_snt

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

    
