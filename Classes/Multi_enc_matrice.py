import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git')
import torch
from Classes.Snt import Snt
from Classes.Matrice_deprecated_2 import Matrice
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

    def tojson(self):
        return {
            "crt": self.crt.tojson(),
            "ctxs": [ctx.tojson() for ctx in self.ctxs],
            "ctxs_heads": [[head.tojson() for head in ctx_heads] for ctx_heads in self.ctxs_heads],
            "sl_heads": [sl_head.tojson() for sl_head in self.sl_heads]
        }

    def suppr_pad(self, padding_mark='<pad>'):
        """Suppression des tokens de padding dans les matrices de chaque contexte.
        """
        # On récupère les positions des tokens de paddings dans la phrase source et de contexte
        list_crt_suppr_pad, list_ctx_suppr_pad = self.sentences_suppr_pad(padding_mark=padding_mark)

        # On supprime les poids correspondant aux tokens de padding dans les têtes token-level
        for k in range(len(self.ctxs)):
            for head in range(len(self.ctxs_heads[k])):
                self.ctxs_heads[k][head] = self.ctxs_heads[k][head].suppr_pad(row_list_suppr_pad= list_crt_suppr_pad, col_list_suppr_pad=  list_ctx_suppr_pad[k])

        # On supprime les poids correspondant aux tokens de padding dans les têtes sentence-level
        for sl_head in range(len(self.sl_heads)):
            self.sl_heads[sl_head] = self.sl_heads[sl_head].suppr_pad(row_list_suppr_pad= list_crt_suppr_pad)

    def fusion_bpe(self, BPE_mark: str = '@@'):
        """Fusion des tokens BPE dans les matrices de chaque contexte.
        """
        list_crt_fusion_bpe, list_ctx_fusion_bpe = self.sentences_fusion_bpe(BPE_mark=BPE_mark)
        # print(f"list_ctx_fusion_bpe: {list_ctx_fusion_bpe}")
        # On fusionne les poids correspondant aux BPEs dans les têtes token-level
        for k in range(len(self.ctxs)):
            for head in range(len(self.ctxs_heads[k])):
                self.ctxs_heads[k][head] = self.ctxs_heads[k][head].fusion_bpe(row_list_groupes= list_crt_fusion_bpe, col_list_groupes=  list_ctx_fusion_bpe[k])

        # On fusionne les poids correspondant aux BPEs dans les têtes sentence-level
        for sl_head in range(len(self.sl_heads)):
            self.sl_heads[sl_head] = self.sl_heads[sl_head].fusion_bpe(row_list_groupes= list_crt_fusion_bpe)

    def clean_matrice(self):
        for k in range(len(self.ctxs)):
            for head in range(len(self.ctxs_heads[k])):
                self.ctxs_heads[k][head].suppr_inf(medium="suppr_inf_uniform")
                self.ctxs_heads[k][head].norm_tenseur(medium="max")
        # for sl_head in range(len(self.sl_heads)):
        #     self.sl_heads[sl_head].norm_tenseur()

    def get_full_ctxs(self) -> Snt:
        """Retourne le contexte sous forme d'une seule Snt.

        Returns:
            Snt: Snt représentant l'ensemble du contexte. L'identifiant correspond à l'identifiant du contexte 
                    le plus éloigné de la phrase courante
        
        Tests:
        >>> crt = Snt(identifiant=3, tokens= ['current1', 'current2', 'current3'])
        >>> ctxs = [Snt(identifiant=1, tokens= ['context1', 'context2', 'context3']), Snt(identifiant=2, tokens= ['context4', 'context5', 'context6'])]
        >>> ctxs_heads = [[Matrice([[1,2,3], [4,5,6], [7,8,9]]), Matrice([[1,2,3], [4,5,6], [7,8,9]])], [Matrice([[1,2,3], [4,5,6], [7,8,9]]), Matrice([[1,2,3], [4,5,6], [7,8,9]])]]
        >>> sl_heads = [Sl_matrice(torch.Tensor([[1,2,3], [1,2,3], [1,2,3]])), Sl_matrice(torch.Tensor([[1,2,3], [1,2,3], [1,2,3]]))]
        >>> m = Multi_enc_matrice(crt=crt, ctxs=ctxs, ctxs_heads=ctxs_heads, sl_heads=sl_heads)
        >>> print(m.get_full_ctxs())
        {'_identifiant': 1, '_tokens': ['context1', 'context2', 'context3', 'context4', 'context5', 'context6']}
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
            # Pour chaque tête du mécanisme sentence_level
            # On récupère une liste des matrices contextualisées des têtes d'attention des mécanismes token-level 
            # dimension contextualised_matrices : nombre de tête token-level x len(phrase courante) x len(phrases de contexte)
            contextualised_matrices = [] 
            if medium == 'full':
                for h_tl in range(len(self.ctxs_heads[0])): 
                    # Pour chaque tête du mécanisme token_level
                    # On insert la matrice contextualisée de la tête d'attention token-level courante 
                    # dans la liste contextualised_matrices
                    contextualised_matrices.append(self.sl_heads[h_sl].contextualise_matrice([ self.ctxs_heads[k][h_tl] for k in range(self.sl_heads[h_sl].size(dim = 1))]))
            elif mean_tl_head is not None:
                contextualised_matrices.append(self.sl_heads[h_sl].contextualise_matrice([ mean_tl_head[k][h_tl] for k in range(self.sl_heads[h_sl].size(dim = 1))]))
            matrices.append(contextualised_matrices)
        return matrices

    def mean_ctxs_heads(self) -> List[Matrice]:
        from Utils import Utils
        mean_ctxs_heads = []
        for k in range(len(self.ctxs)):
            mean_ctxs_heads.append(Matrice(Utils.mean_matrices([self.ctxs_heads[k][head].matrice for head in range(len(self.ctxs_heads[k])) ])))
        return mean_ctxs_heads

    def mean_sl_heads(self) -> Sl_matrice:
        from Utils import Utils
        return Sl_matrice(Utils.mean_matrices([self.sl_heads[sl_head].matrice for sl_head in range(len(self.sl_heads))]))

    def ecriture_xlsx(self, absolute_folder, filename= None, precision: int = 2, create_folder_path= False):
        """Ecriture des matrices dans un fichier Excel.

        Args:
            absolute_folder (str): Répertoire absolu pour l'écriture du fichier Excel.
            filename (str, optional): Nom du fichier Excel. Par défaut, ctx_{k}_head_{head}.
            create_folder_path (bool, optional): Créer le répertoire s'il n'existe pas. Par défaut, False.
        """
        for k in range(len(self.ctxs_heads)):
            for head in range(len(self.ctxs_heads[k])):
                self.ctxs_heads[k][head].ecriture_xlsx(crt= self.crt,
                                        ctx= self.ctxs[k],
                                        absolute_folder= f"{absolute_folder}/{head}",
                                        filename=f"{filename}_k{len(self.ctxs) - k}_h{head}" if filename else f"ctx_{k}_head_{head}",
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
    from Utils import Utils_multi_enc

    torch.set_printoptions(precision=2)
    print(f"[DEBUG] Doctest clear")
    _DEBUG_START = True
    _DEBUG_SUPPR_PAD = True
    _DEBUG_NORM_TENSOR = True
    _DEBUG_FUSION_BPE= True
    _PRECISION = 3
    _OUTPUT_PATH=f"/home/getalp/lopezfab/Documents"
    id = 1850

    r_path=f"/home/getalp/lopezfab/lig/temp/temp/temp/han_attn2/{id}.json"
    data=Utils_data.lecture_data(r_path)
    crt, ctxs, ctxs_heads, sl_heads = Utils_data.lecture_multi_enc_objet(data)
    # print(f"[debug] m1/sl_heads: {sl_heads}")
    m1 = Utils_multi_enc.pre_traitement_src(crt, ctxs, sl_heads, ctxs_heads)
    if _DEBUG_START:
        m1.ecriture_xlsx(absolute_folder=f"/home/getalp/lopezfab/Documents/{id}/test_contextualised",
                            filename="raw",
                            precision=6,
                            create_folder_path=True)
    
    m1.suppr_pad()
    if _DEBUG_SUPPR_PAD:
        m1.ecriture_xlsx(absolute_folder=f"{_OUTPUT_PATH}/{id}/test_contextualised",
                            filename="suppr_pad",
                            precision=_PRECISION,
                            create_folder_path=True)
    
    m1.clean_matrice()
    if _DEBUG_NORM_TENSOR:
        m1.ecriture_xlsx(absolute_folder=f"{_OUTPUT_PATH}/{id}/test_contextualised",
                            filename="clean_matrice",
                            precision=_PRECISION,
                            create_folder_path=True)
    # print(f"[debug] m1/sl_heads: {m1.sl_heads}")
    m1.fusion_bpe()
    if _DEBUG_FUSION_BPE:
        m1.ecriture_xlsx(absolute_folder=f"{_OUTPUT_PATH}/{id}/test_contextualised",
                            filename="fusion_bpe",
                            precision= _PRECISION,
                            create_folder_path=True)
    
    
    test = m1.get_crt_to_ctxs('full')
    # test List[List[Matrice]]. Taille : nb_sl_heads x nb_tl_heads x [crt x ctxs]
    print(f"crt len vs. test crt len: {len(m1.crt)} vs. {test[0][0].size(dim = 0)}")
    print(f"ctx len vs. test ctx len: {len(m1.get_full_ctxs())} vs. {test[0][0].size(dim = 1)}")
    for sl_heads in range(len(test)):
        for tl_heads in range(len(test[sl_heads])):
            test[sl_heads][tl_heads].norm_tenseur()
            test[sl_heads][tl_heads].ecriture_xlsx(crt = m1.crt, ctx = m1.get_full_ctxs(), absolute_folder= f"{_OUTPUT_PATH}/{id}/test_contextualised/{sl_heads}", filename = f"{tl_heads}", create_folder_path=True)

