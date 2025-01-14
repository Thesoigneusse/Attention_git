import torch
from typing import List
from copy import deepcopy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Classes.Snt import Snt
    from Classes.Matrice import Matrice
# N : taille de la phrase courante
# k : nombre de phrase de contexte
# M_k : Taille de la phrase de contexte k
# M : taille des phrases de contexte
# S : Taille de la fusion des phrases de contexte
# nb_heads : nombre de tête d'attention
# L : nombre de layers



def ajoute_eos_tokens_src(_snt: list, src_segments_labels: list, eos_token: str = "<eos>") -> list:
    """Ajoute un token end-of-sentence dans la phrase afin de faire concorder la taille de la matrice et de la phrase

    Args:
        _snt (list): list de token de la phrase côté source
        src_segments_labels (list): list d'id de contexte pour chaque token de la phrase _snt
        eos_token (str, optional): token end-of-sentence. Defaults to "<eos>".

    Returns:
        list: list de tokens de la phrase coté source avec les tokens end-of-sentence
    """
    assert len(_snt) <= len(src_segments_labels), f"[DEBUG]Longueur error, len(snt) vs. len(labels): {len(_snt)} vs. {len(src_segments_labels)}"
    snt = deepcopy(_snt)
    # Si on a des éléments à ajouter
    if len(snt) < len(src_segments_labels):
        # Parcours de src_segments_label
        for i in range(len(src_segments_labels) - 1):
            # Si on trouve un changement de segments alors on ajoute un token <eos>
            if src_segments_labels[i] != src_segments_labels[i+1]:
                snt.insert(i, eos_token)
    assert len(snt) == len(src_segments_labels)
    return snt

def full_sentence_to_ctx_and_crt(_snt: 'Snt', eos_token: str = "<eos>"):
    """Découpe une liste de tokens contenant plusieurs phrases en une liste de phrase où chaque phrase correspond à une liste de tokens

    Args:
        _snt (list): list de tokens de phrases
        eos_token (str, optional): Token end-of-sentence. Defaults to "<eos>".

    Returns:
        list: liste de phrases
    >>> s1 = Snt(identifiant = 3, tokens= ["t7", "t6", "<eos>", "t5", "t4", "<eos>", "t3", "t2", "<eos>", "t1", "t0", "<END>"])
    >>> print(full_sentence_to_ctx_and_crt(s1))
    [{'_identifiant': 0, '_tokens': ['t7', 't6', '<eos>']}, {'_identifiant': 1, '_tokens': ['t5', 't4', '<eos>']}, {'_identifiant': 2, '_tokens': ['t3', 't2', '<eos>']}, {'_identifiant': 3, '_tokens': ['t1', 't0', '<END>']}]
    """
    from Classes.Snt import Snt
    # On parcours la phrase complète et on ajoute 
    index_eos = [ i for i, tok in enumerate(_snt.tokens) if tok == eos_token]
    if len(index_eos) > 0:
        snt = deepcopy(_snt)
        list_sentence = []
        start = 0
        for k, index in enumerate(index_eos):
            list_sentence.append(Snt(identifiant= snt.identifiant - len(index_eos) + k, tokens= snt.tokens[start:index]))
            list_sentence[k].append(eos_token)# Correction du eos_token manquant en fin de phrase
            start = index + 1
        list_sentence.append(Snt(identifiant= snt.identifiant, tokens= snt.tokens[start:]))
        return list_sentence


def correctif_src_sentence(src: List[str], src_seg_lab: List[int]) -> None:
    """Corrige les phrases d'entrée côté source où il manque les tokens <eos> dans 
    la phrase concaténée alors qu'ils sont présent dans la variable src_seg_lab

    Args:
        src (List[str]): liste of the token of the source sentence.
        src_seg_lab (List[int]): Liste de l'emplacement  de phrase du token correspondant dans la variable src.
        0 : phrase courante
        1 : phrase de contexte précédente,
        etc 
    """
    
    for itok in range(len(src_seg_lab) -1):
        if int(src_seg_lab[itok]) != int(src_seg_lab[itok +1]):
            src.insert(itok + 1, '<eos>')        
        itok += 1
    return src 


def cut_matrix_into_sentences(_matrice: 'Matrice', snts: List[str]) -> List[List['Matrice']]:
    """Découpe une Matrice ctx+crt de self-attention en différentes Matrices k3 x k3, k3 x k2, ..., k0 x k0

        k3xk3   k3xk2   k3xk1   k3xk0
        k2xk3   k2xk2   k2xk1   k2xk0
        k1xk3   k1xk2   k1xk1   k1xk0
        k0xk3   k0xk2   k0xk1   k0xk0
    

    Args:
        _matrice (Matrice): Matrice de self-attention ctx+crt x ctx+crt
        snts (List[Snt]): liste de Snt

    Returns:
        list: liste de liste de Matrice

    >>> liste_10_x_10 = [[0 , 1, 2, 3, 4, 5, 6, 7, 8, 9], [10,11,12,13,14,15,16,17,18,19], [20,21,22,23,24,25,26,27,28,29], [30,31,32,33,34,35,36,37,38,39], [40,41,42,43,44,45,46,47,48,49], [50,51,52,53,54,55,56,57,58,59], [60,61,62,63,64,65,66,67,68,69], [70,71,72,73,74,75,76,77,78,79], [80,81,82,83,84,85,86,87,88,89], [90,91,92,93,94,95,96,97,98,99]]
    >>> snts = [Snt(identifiant=0, tokens = ["k3t1", "k3t2"]), Snt(identifiant=1, tokens = ["k2t1", "k2t2", "k2t3"]), Snt(identifiant=2, tokens = ["k1t1", "k1t2"]), Snt(identifiant=3, tokens = ["k0t1", "k0t2", "k0t3"])]
    >>> m1 = Matrice(torch.Tensor(liste_10_x_10))
    >>> print(cut_matrix_into_sentences(m1, snts))
    [[{'_matrice': tensor([[ 0.,  1.],
            [10., 11.]])}, {'_matrice': tensor([[ 2.,  3.,  4.],
            [12., 13., 14.]])}, {'_matrice': tensor([[ 5.,  6.],
            [15., 16.]])}, {'_matrice': tensor([[ 7.,  8.,  9.],
            [17., 18., 19.]])}], [{'_matrice': tensor([[20., 21.],
            [30., 31.],
            [40., 41.]])}, {'_matrice': tensor([[22., 23., 24.],
            [32., 33., 34.],
            [42., 43., 44.]])}, {'_matrice': tensor([[25., 26.],
            [35., 36.],
            [45., 46.]])}, {'_matrice': tensor([[27., 28., 29.],
            [37., 38., 39.],
            [47., 48., 49.]])}], [{'_matrice': tensor([[50., 51.],
            [60., 61.]])}, {'_matrice': tensor([[52., 53., 54.],
            [62., 63., 64.]])}, {'_matrice': tensor([[55., 56.],
            [65., 66.]])}, {'_matrice': tensor([[57., 58., 59.],
            [67., 68., 69.]])}], [{'_matrice': tensor([[70., 71.],
            [80., 81.],
            [90., 91.]])}, {'_matrice': tensor([[72., 73., 74.],
            [82., 83., 84.],
            [92., 93., 94.]])}, {'_matrice': tensor([[75., 76.],
            [85., 86.],
            [95., 96.]])}, {'_matrice': tensor([[77., 78., 79.],
            [87., 88., 89.],
            [97., 98., 99.]])}]]
    """
    from Classes.Matrice import Matrice

    assert _matrice.matrice.size(dim=0) == sum([len(snt) for snt in snts]), f"[DEBUG]Size error, matrice size dim0 vs. snt len: {_matrice.matrice.size()} vs. {sum([len(snt) for snt in snts])}"
    assert _matrice.matrice.size(dim=1) == sum([len(snt) for snt in snts]), f"[DEBUG]Size error, matrice size dim1 vs. snt len: {_matrice.matrice.size()} vs. {sum([len(snt) for snt in snts])}"

    debut_row, fin_row = 0, 0
    matrices = []
    # Pour chaque ligne
    for i in range(len(snts)):
        fin_row += len(snts[i])
        debut_col, fin_col = 0, 0
        # temp: liste temporaire
        temp = []
        # Pour chaque colonne
        for j in range(len(snts)):
            fin_col += len(snts[j])
            # On ajoute les éléments de la matrice dans la liste temporaire
            temp.append(Matrice(_matrice.matrice[
                    debut_row:fin_row,
                    debut_col:fin_col
                ]))
            debut_col = fin_col
        matrices.append( deepcopy(temp))
        debut_row = fin_row
    return matrices

def pre_traitement_src(identifiant: int, matrices: List[List[torch.Tensor]], full_snt: List[str], ssl  : List[str]):
    from Classes.Snt import Snt
    from Classes.Matrice import Matrice
    # Correction du token <eos> manquant
    full_snt = Snt(identifiant= identifiant, tokens = ajoute_eos_tokens_src(_snt= full_snt.split(), src_segments_labels=ssl))
    snt_cutted = full_sentence_to_ctx_and_crt(full_snt)

    # Au moins une phrase de contexte et la phrase courante (+ 1 car contient une phrase vide quand le nb de contexte est inférieur à la normale)
    if len(snt_cutted) > 2: 
        # Extraction des phrases de contexte et de la phrase courante
        ctxs = []
        for k in range(len(snt_cutted[:-1])):
            ctxs.append(snt_cutted[k])
        
        # Extraction des différentes matrices à travers les 6 layers et les 8 têtes d'attention de chaque layer
        layers = []
        for layer in range(len(matrices)): # Pour chaque layer
            heads = []
            for head in range(len(matrices[layer])): # on extrait chaque tête par layer
                full_matrice = torch.tensor(matrices[layer][head])
                full_matrice = full_matrice.squeeze() # on supprime une dimension qui semble inutile (=1)
                heads.append(Matrice(full_matrice))
            layers.append(heads)
        # layers : L x [nb_heads x [torch.Tensor(N x N)]]

        # Traitement du cas particulier où un contexte n'est pas présent. Suppression des reliquats dans le contexte et les matrices d'attention
        for i in range(len(ctxs)-1, -1, -1):
            if ctxs[i].tokens == ["<eos>"]:
                del ctxs[i]
                del snt_cutted[i]
                for layer in range(len(layers)):
                    for head in range(len(layers[layer])):
                        layers[layer][head].matrice = torch.cat([layers[layer][head].matrice[1:, 1:]])

        full_ctx = ctxs[0].copy() # Snt(identifiant= identifiant - len(ctxs), tokens= ctxs[0].tokens)
        if len(ctxs) > 1:
            for snt in ctxs[1:]:
                full_ctx += snt
                # print(f"[debug] len(full_ctx): {len(full_ctx)}")
        crt = snt_cutted[-1].copy()
        # if _FULL_SNT: # permet prendre en compte la phrase courante ou non
        #     full_ctx.tokens += crt.tokens

        cutted_layers = []
        for layer in range(len(layers)):
            cutted_heads = []
            for head in range(len(layers[layer])):
                cutted_heads.append(cut_matrix_into_sentences(layers[layer][head], snt_cutted))
                # Dernière liste correspond à la phrase courante vers les phrases de contexte et la phrase courante
                # _FULL_SNT permet prendre en compte la phrase courante ou non
                # if _FULL_SNT:
                #     layers[layer][head].matrice = torch.cat([matrice.matrice for matrice in full_matrice_cutted[-1][:]], dim = 1)
                # else:
                #     layers[layer][head].matrice = torch.cat([matrice.matrice for matrice in full_matrice_cutted[-1][:-1]], dim = 1)
            cutted_layers.append(cutted_heads)

    return {'identifiant': identifiant, # int permettant d'identifier la sentence
            'snt_cutted': snt_cutted, # List[S_k]
            'full_ctx': full_ctx, # M
            'crt': crt, # N
            'layers': layers, # L[ nb_heads[ M * M ] ]
            'cutted_layers': cutted_layers,} # L[ nb_heads[ k+crt[ k+crt[ len(S_k) x len(S_k) ] ] ] ]

# def pre_traitement_tgt(identifiant: int, matrices: List[List[torch.Tensor]])


if __name__ == '__main__':
    import doctest
    import sys
    sys.path.append("/home/getalp/lopezfab/Bureau/Attention_git/")
    from Classes.Snt import Snt
    from Classes.Matrice import Matrice
    doctest.testmod()
    print(f"[DEBUG] Doctest clear")

    # s1 = Snt(identifiant = 3, tokens= ["t7", "t6", "<eos>", "t5", "t4", "<eos>", "t3", "t2", "<eos>", "t1", "t0", "<END>"])
    # print(full_sentence_to_ctx_and_crt(s1))

