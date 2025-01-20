from Classes.Snt import Snt
from Classes.Matrice import Matrice
from Classes.CA_matrice import CA_matrice
from Classes.Sl_matrice import Sl_matrice
from typing import List
# N : taille de la phrase courante
# k : nombre de phrase de contexte
# M_i : Taille de la ième phrase de contexte 
# M : Taille de la fusion des phrases de contexte
# nb_heads : nombre de tête d'attention
# L : nombre de layers

class Multi_enc_matrice(CA_matrice):
    def __init__(self, crt: Snt = None, ctxs: List[Snt] = None, ctxs_heads: List[Matrice] = None, sl_heads: List[Sl_matrice] = None) -> None:
        super().__init__(crt=crt, ctxs=ctxs)

        # Variable correspondant aux têtes du mécanisme d'attention sentence_level
        # Dimension : nb_heads x N x k
        self.sl_heads = sl_heads
        
        # Variable correspondant aux têtes du mécanisme d'attention token-level
        # Dimension : k x nb_head x N x M_i
        self.ctxs_heads = ctxs_heads
    
    @property
    def sl_heads(self) -> List[Sl_matrice]:
        # Dimension : nb_heads x N x k
        return self._sl_heads
    @sl_heads.setter
    def sl_heads(self, value: List[Sl_matrice]) -> None:
        if not isinstance(value, type(None)):
            assert isinstance(value, list), f"sl_heads must be a list. Current Value: {type(value)}"
            assert all(isinstance(sl_matrice, Sl_matrice) for sl_matrice in value), f"sl_heads must be a list of Sl_matrice. Current Value: {[type(sl_matrice) for sl_matrice in value]}"
        self._sl_heads = value

    @property
    def ctxs_heads(self) -> List[List[Matrice]]:
        # Dimension : k x nb_head x N x M
        return self._ctxs_heads
    @ctxs_heads.setter
    def ctxs_heads(self, value: List[List[Matrice]]) -> None:
        if not isinstance(value, type(None)):
            assert isinstance(value, list), f"ctxs must be a list. Current Value: {type(value)}"
            assert all(isinstance(head, List) for head in value), f"ctxs must be a list of list. Current Value: {[type(head) for head in value]}"
            assert all(all(isinstance(matrice, Matrice) for matrice in head) for head in value), f"ctxs must be a list of list of Matrice. Current Value: {[type(head[0]) for head in value]}"
        self._ctxs_heads = value

    def suppr_pad(self, padding_mark='<pad>'):
        """Suppression des tokens de padding dans les matrices de chaque contexte.
        """
        # On récupère les positions des tokens de paddings dans la phrase source et de contexte
        list_crt_suppr_pad, list_ctx_suppr_pad = self.sentences_suppr_pad(padding_mark=padding_mark)

        # On supprime les poids correspondant aux tokens de padding dans les têtes token-level
        for k in range(len(self.ctxs)):
            for head in range(len(self.ctxs_heads[k])):
                self.ctxs_heads[k][head].suppr_pad(row_list_suppr_pad= list_crt_suppr_pad, col_list_suppr_pad=  list_ctx_suppr_pad[k])

        # On supprime les poids correspondant aux tokens de padding dans les têtes sentence-level
        for sl_head in range(len(self.sl_heads)):
            self.sl_heads[sl_head].suppr_pad(row_list_suppr_pad= list_crt_suppr_pad)

    def fusion_bpe(self, BPE_mark: str = '@@'):
        """Fusion des tokens BPE dans les matrices de chaque contexte.
        """
        list_crt_fusion_bpe, list_ctx_fusion_bpe = self.sentences_fusion_bpe(BPE_mark=BPE_mark)

        # On fusionne les poids correspondant aux BPEs dans les têtes token-level
        for k in range(len(self.ctxs)):
            for head in range(len(self.ctxs_heads[k])):
                self.ctxs_heads[k][head].fusion_bpe(row_list_fusion_bpe= list_crt_fusion_bpe, col_list_fusion_bpe=  list_ctx_fusion_bpe[k])

        # On fusionne les poids correspondant aux BPEs dans les têtes sentence-level
        for sl_head in range(len(self.sl_heads)):
            self.sl_heads[sl_head].fusion_bpe(row_list_fusion_bpe= list_crt_fusion_bpe)

    def clean_matrice(self):
        for k in range(len(self.ctxs)):
            for head in range(len(self.ctxs_heads[k])):
                self.ctxs_heads[k][head].suppr_inf(medium="inf_uniform")
                self.ctxs_heads[k][head].norm_tensor(medium="max")
        for sl_head in range(len(self.sl_heads)):
            self.sl_heads[sl_head].norm_tensor()

    def get_full_ctxs(self) -> Snt:
        """Retourne le contexte sous forme d'une seule Snt.

        Returns:
            Snt: Snt représentant l'ensemble du contexte. L'identifiant correspond à l'identifiant du contexte 
                    le plus éloigné de la phrase courante
        """
        ctxs = Snt(identifiant=self.crt.identifiant, tokens= [])
        for k in range(len(self.ctxs)):
            ctxs += self.ctxs[k]
            ctxs.identifiant -= 1
        return ctxs

    def get_crt_to_ctxs(self, medium: str = 'full'):
        """Retourne une Matrice entre la phrase courante et les phrases de contextes

        Returns:
            Matrice: Matrice entre la phrase courante et les phrases de contextes
        """
        matrices =  []
        mean_tl_head = self.mean_ctxs_heads() if medium == 'tl_mean' else None
        mean_sl_head = self.mean_sl_heads() if medium == 'sl_mean' else None
        
        for h_sl in range(len(self.sl_heads)):
            contextualised_matrices = []
            if medium == 'full':
                for h_tl in range(len(self.ctxs_heads[0])):
                    contextualised_matrices.append(self.sl_heads[h_sl].contextualise_matrice([ self.ctxs_heads[k][h_tl] for k in range(len(self.sl_heads[h_sl].size(dim = 1)))]))
            elif mean_tl_head is not None:
                contextualised_matrices.append(self.sl_heads[h_sl].contextualise_matrice([ mean_tl_head[k][h_tl] for k in range(len(self.sl_heads[h_sl].size(dim = 1)))]))
            matrices.append(contextualised_matrices)
        return Matrice(matrices)

    def mean_ctxs_heads(self) -> List[Matrice]:
        from Utils import Utils
        mean_ctxs_heads = []
        for k in range(len(self.ctxs)):
            mean_ctxs_heads.append(Matrice(Utils.mean_matrices([self.ctxs_heads[k][head].matrice for head in range(len(self.ctxs_heads[k])) ])))
        return mean_ctxs_heads

    def mean_sl_heads(self) -> Sl_matrice:
        from Utils import Utils
        return Sl_matrice(Utils.mean_matrices([self.sl_heads[sl_head].matrice for sl_head in range(len(self.sl_heads))]))

    def ecriture_xslx(self, absolute_folder, filename= None, precision: int = 2, create_folder_path= False):
        """Ecriture des matrices dans un fichier Excel.

        Args:
            absolute_folder (str): Répertoire absolu pour l'écriture du fichier Excel.
            filename (str, optional): Nom du fichier Excel. Par défaut, ctx_{k}_head_{head}.
            create_folder_path (bool, optional): Créer le répertoire s'il n'existe pas. Par défaut, False.
        """
        for k in range(len(self.ctxs_heads)):
            for head in range(len(self.ctxs_heads[k])):
                Matrice.ecriture_xslx(matrice=self.ctxs_heads[k][head].matrice,
                                        crt= self.crt,
                                        ctx= self.ctxs[k],
                                        absolute_folder= f"{absolute_folder}/{head}",
                                        filename=f"{filename}_k{k}_h{head}" if filename else f"ctx_{k}_head_{head}",
                                        precision=precision,
                                        create_folder_path=create_folder_path)

    def test_multi_enc_matrice(self):
        super().test_()
        sl_m = Sl_matrice()
        sl_m.test_([10, 3])
        self.sl_heads = sl_m*8
        m = Matrice()
        m.test_()
        self.ctxs_heads = [m*8]*3

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    import torch
    from Utils import Utils_data

    torch.set_printoptions(precision=2)
    print(f"[DEBUG] Doctest clear")
    _DEBUG_START = True
    _DEBUG_SUPPR_PAD = True
    _DEBUG_NORM_TENSOR = True
    _DEBUG_FUSION_BPE= False
    _PRECISION = 3
    _OUTPUT_PATH=f"/home/getalp/lopezfab/Documents"
    id = 1850
    r_path=f"/home/getalp/lopezfab/lig/temp/temp/temp/han_attn2/{id}.json"
    data=Utils_data.lecture_data(r_path)
    crt, ctxs, ctxs_heads, sl_heads = Utils_data.lecture_objet(data)
    m1 = Multi_enc_matrice(crt=crt,
                            ctxs= ctxs,
                            ctxs_heads=ctxs_heads,
                            sl_heads=sl_heads)
    if _DEBUG_START:
        m1.ecriture_xslx(absolute_folder=f"/home/getalp/lopezfab/Documents/{id}",
                            filename="raw",
                            precision=6,
                            create_folder_path=True)
    m1.suppr_pad()
    if _DEBUG_SUPPR_PAD:
        m1.ecriture_xslx(absolute_folder=f"{_OUTPUT_PATH}/{id}",
                            filename="suppr_pad",
                            precision=_PRECISION,
                            create_folder_path=True)
    
    m1.clean_matrice()
    if _DEBUG_NORM_TENSOR:
        m1.ecriture_xslx(absolute_folder=f"{_OUTPUT_PATH}/{id}",
                            filename="clean_matrice",
                            precision=_PRECISION,
                            create_folder_path=True)

    m1.fusion_bpe()
    if _DEBUG_FUSION_BPE:
        m1.ecriture_xslx(absolute_folder=f"{_OUTPUT_PATH}/{id}",
                            filename="fusion_bpe",
                            precision= _PRECISION,
                            create_folder_path=True)

