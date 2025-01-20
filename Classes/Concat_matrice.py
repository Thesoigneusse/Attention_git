import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Classes.Matrice import Matrice
from Classes.CA_matrice import CA_matrice
from Classes.Snt import Snt
from Utils import Utils_concat
from typing import List
import torch
from copy import copy
# N : taille de la phrase courante
# k : nombre de phrase de contexte
# ctx_k : Taille de la phrase de contexte k
# ctxs : Phrase de contexte sous forme de List[Snt]
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
        from Utils import Utils
        # Fusion des bpes pour les phrases courante et de contextes
        full_snt = self.get_full_snt()
        snt_list_fusion_bpe = self.get_full_snt().fusion_bpe()

        row_list_fusion_bpe = self.crt.fusion_bpe()
        col_list_fusion_bpe = []
        for k in range(len(self.ctxs)):
            col_list_fusion_bpe.append(self.ctxs[k].fusion_bpe())

        # Fusion des bpes pour les matrices d'attention de chaque layer et de chaque head
        # print(f"[DEBUG] full_snt: {full_snt}")
        # print(f"[DEBUG] snt_list_fusion_bpe: {snt_list_fusion_bpe}")
        assert len(full_snt) == self.layers[0][0].size(dim=0), f'taille non compatible. len(full_snt) vs. len(self.layers): {len(full_snt)} vs. {self.layers[0][0].size(dim = 0)}'
        assert len(full_snt) == self.layers[0][0].size(dim=1), f'taille non compatible. len(full_snt) vs. len(self.layers): {len(full_snt)} vs. {self.layers[0][0].size(dim = 1)}'

        for i_layer in range(len(self.layers)):
            for h_head in range(len(self.layers[i_layer])):
                self.layers[i_layer][h_head] = self.layers[i_layer][h_head].fusion_bpe(groupes=snt_list_fusion_bpe)
                self.layers[i_layer][h_head] = self.layers[i_layer][h_head].transpose(0,1).fusion_bpe(groupes=snt_list_fusion_bpe).transpose(0,1)
        return {'crt': row_list_fusion_bpe, 'ctxs': col_list_fusion_bpe}

    def norm_tensor(self, data_to_norm= "cutted_matrices", medium = "minmax") -> None:
        """Applique une normalisation sur les matrices des chaque layers et de chaque tête d'attention

        Args:
            medium (str, optional): médium de normalisation. Defaults to "minmax".
        """
        # Normalisation pour les matrices d'attention de chaque layer et de chaque head
        for i_layer in range(len(self.layers)):
            for h_head in range(len(self.layers[i_layer])):
                self.layers[i_layer][h_head].norm_tensor(medium=medium)

    def get_full_snt(self, to_process: str = None):
        full_snt = Snt(identifiant=self.crt.identifiant, tokens=[])
        for ctx in self.ctxs:
            full_snt += ctx
            full_snt.identifiant -= 1
        if to_process is None or to_process == 'crt_to_ctxs_crt':
            full_snt += self.crt
        return full_snt

    def get_snts(self) -> List['Snt']:
        """Retourne une liste de Snt (phrases de contexte + phrase courante)
           Taille: K+1

        Returns:
            List['Snt']: Liste des phrases de contexte + phrase courante
        """
        from copy import deepcopy
        snts = deepcopy(self.ctxs)
        snts.append(self.crt)
        return snts

    def get_head_means(self, layer = None) -> List[Matrice]:
        """Retourne la liste des têtes d'attention moyennée par layers
            L x nb_heads x N x M -> L x N x M

        Args:
            layer (int, optional): layer spécifique à retourner. Si None retourne la liste des layers. Defaults to None.

        Returns:
            List[Matrice]: Liste du/des layers moyennés
        
        Tests:

        """
        import torch
        temp_layers = []
        if layer is None:
            for l in range(len(self.layers)):
                temp_layers.append(Matrice(torch.mean(torch.stack(self.layers[l], dim = -1), dim = -1)))
        elif isinstance(layer, int):
            temp_layers.append(Matrice(torch.mean(torch.stack(self.layers[layer], dim = -1), dim = -1)))
        return temp_layers

    def get_crt_to_ctxs_crt(self, layer: int = None, head: int = None) -> List[List[Matrice]]:
        """Retourne une liste de liste de Matrice objects

        Args:
            layer (int, optional): index du layer à traiter. None traite tous les layers. Defaults to None.
            head (int, optional): index de la head à traiter. None traite toutes les heads. Defaults to None.

        Returns:
            List[List[Matrice]]: Liste de Liste des Matrices à traiter
        """
        snts = self.get_snts()
        matrices = []
        if layer is None and head is None:
            # Retourne une liste de liste de matrice 
            # Taille: L[ H[ crt x (ctxs+crt ] ]
            for l in range(len(self.layers)):
                matrices_layer = []
                for h in range(len(self.layers[l])):
                    matrices_layer.append(Matrice(torch.cat(Utils_concat.cut_matrix_into_sentences(_matrice = self.layers[l][h], snts= snts)[-1], dim = 1)))
                matrices.append(matrices_layer)
        elif isinstance(layer, int) and head is None:
            # Retourne une liste de liste de matrice 
            # Taille: L[ mean_head[ crt x (ctxs+crt ] ]
            matrices.append([Matrice(torch.cat(Utils_concat.cut_matrix_into_sentences(_matrice = self.get_head_means(layer=layer)[0], snts= snts)[-1], dim = 1))])
        elif isinstance(layer, int) and isinstance(head, int):
            # Retourne une liste de liste de matrice 
            # Taille: layer[ head[ crt x (ctxs+crt ] ]
            matrices.append([Matrice(torch.cat(Utils_concat.cut_matrix_into_sentences(_matrice = self.layers[layer][head], snts= snts)[-1], dim = 1))])
        return matrices

    def get_crt_to_ctxs(self) -> List[List[Matrice]]:
        snts = self.get_snts()
        matrices = []
        for l in range(len(self.layers)):
            matrice_layer = []
            for h in range(len(self.layers[l])):
                matrice_layer.append(Matrice(torch.cat(Utils_concat.cut_matrix_into_sentences(_matrice = self.layers[l][h], snts= snts)[-1][:-1], dim = 1)))
            matrices.append(matrice_layer)
        return matrices

    def ecriture_xslx(self, data_to_write = "cutted_matrices", absolute_folder: str = None, precision: int = 2, create_folder_path: bool = False) -> None:
        # écriture de la full matrice
        if data_to_write == 'full_matrice' or data_to_write == 'all':
            # Pour chaque layer
            for l in range(len(self.layers)):
                # Pour chaque head
                for h in range(len(self.layers[l])):
                    # on écrit la matrice layer/head correspondant
                    self.layers[l][h].ecriture_xslx(crt= self.get_full_snt(),
                                                    ctx= self.get_full_snt(),
                                                    absolute_folder= f"{absolute_folder}/full_matrice/{l}",
                                                    filename = f"{h}",
                                                    precision = precision,
                                                    create_folder_path = create_folder_path)
                # Puis pour chaque layer on écrit la moyenne des heads
                self.get_head_means(layer=l)[0].ecriture_xslx(crt= self.get_full_snt(),
                                                              ctx= self.get_full_snt(),
                                                              absolute_folder= f"{absolute_folder}/full_matrice/{l}",
                                                              filename = f"mean",
                                                              precision = precision,
                                                              create_folder_path = create_folder_path)
        # écriture de la matrice pour la phrase courante vers toutes les phrases (ctxs + crt)
        snts = self.get_snts()
        if data_to_write == 'crt_to_full_ctxs_crt' or data_to_write == 'all':
            # Pour chaque layer
            for l in range(len(self.layers)):
                # Pour chaque head
                for h in range(len(self.layers[l])):
                    # on écrit la matrice crt x (ctxs+crt) pour chaque layer/head
                    matrices = Matrice(torch.cat(Utils_concat.cut_matrix_into_sentences(_matrice = self.layers[l][h], snts= snts)[-1], dim = 1))
                    ctx = self.get_full_snt()
                    matrices.ecriture_xslx(crt= self.crt, ctx=ctx , absolute_folder= f"{absolute_folder}/crt_to_full_ctxs_crt/{l}", filename = f"{h}", precision = precision, create_folder_path = create_folder_path)
                # Puis pour chaque layer on écrit la moyenne des heads pour crt x (ctxs+crt)
                matrice = Matrice(torch.cat(Utils_concat.cut_matrix_into_sentences(_matrice = self.get_head_means(layer=l)[0], snts= snts)[-1], dim = 1))
                matrice.ecriture_xslx(crt= self.crt, ctx= self.get_full_snt(), absolute_folder= f"{absolute_folder}/crt_to_full_ctxs_crt/{l}", filename = f"mean", precision = precision, create_folder_path = create_folder_path)
        # écriture des différents contextes séparémment
        if data_to_write == 'cutted_matrices' or data_to_write == 'all':
            # Pour chaque layer
            for l in range(len(self.layers)):
                # Pour chaque tête
                for h in range(len(self.layers[l])):
                    # On découpe la matrice en fonction des ctxs et crt
                    matrices = Utils_concat.cut_matrix_into_sentences(_matrice = self.layers[l][h], snts= snts)
                    # On récupère et écrit les matrices crt x ctx_k
                    for k in range(len(matrices[-1])): # 
                        matrices[-1][k].ecriture_xslx(crt= self.crt, ctx=self.ctxs[k] ,absolute_folder= f"{absolute_folder}/cutted/{l}/{h}", filename = f"{len(matrices[-1]) - k - 1}", precision = precision, create_folder_path = create_folder_path)
                # Puis on découpe la matrice pour chaque moyenne de tête
                matrice = Utils_concat.cut_matrix_into_sentences(_matrice = self.get_head_means(layer=l)[0], snts= snts)
                # Et on écrit les matrices moyennées entre crt et ctx_k
                for k in range(len(matrice[-1])):
                    matrice[-1][k].ecriture_xslx(crt= self.crt, ctx=self.ctxs[k] ,absolute_folder= f"{absolute_folder}/cutted/{l}/mean", filename = f"{len(matrices[-1]) - k - 1}", precision = precision, create_folder_path = create_folder_path)


if __name__ == '__main__':
    import doctest
    import torch

    from Utils import Utils_data
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
    # r_path=f"/home/getalp/lopezfab/lig/temp/temp/test_attn/{id}.json"
    # data=Utils_data.lecture_data(r_path)
    # crt, ctxs, ctxs_heads, sl_heads = Utils_data.lecture_concat_objet(data)
    # m1 = Concat_matrice(crt= Snt().test_(),
    #                     ctxs= [Snt().test_()],
    #                     layers=[[Matrice().test_(), Matrice().test_(10), Matrice().test_(20)]])
    # test = m1.get_head_means()
    # print(test)
    
