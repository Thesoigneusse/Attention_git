from typing import List
from copy import deepcopy
from Matrice import Matrice
from Snt import Snt

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

def full_sentence_to_ctx_and_crt(_snt: list, eos_token: str = "<eos>"):
    """Découpe une liste de tokens contenant plusieurs phrases en une liste de phrase où chaque phrase correspond à une liste de tokens

    Args:
        _snt (list): list de tokens de phrases
        eos_token (str, optional): Token end-of-sentence. Defaults to "<eos>".

    Returns:
        list: liste de phrases
    """
    snt = deepcopy(_snt)
    list_sentence = [[]]
    for i in range(len(snt.tokens)): # On parcourt la liste des tokens
        if snt.tokens[i] == eos_token: 
            # Si on rencontre un token <eos> alors on l'ajoute puis créé une autre phrase
            list_sentence[-1].append(eos_token)
            list_sentence.append([])
        else:
            # Sinon on ajoute le token courant à la denière phrase
            list_sentence[-1].append(snt.tokens[i])
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


def cut_matrix_into_sentences(_matrice: Matrice, snts: List[str]) -> List[List[Matrice]]:
    """Découpe une Matrice ctx+crt de self-attention en différentes Matrices k3 x k3, k3 x k2, ..., k0 x k0

    Args:
        _matrice (Matrice): Matrice de self-attention ctx+crt x ctx+crt
        snts (List[Snt]): liste de Snt

    Returns:
        list: liste de liste de Matrice
    """
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
            temp.insert(0, {
                "crt": deepcopy(snts[i]),
                "ctx": deepcopy(snts[j]),
                "matrix": Matrice(deepcopy(_matrice.matrice[
                    debut_row:fin_row,
                    debut_col:fin_col
                ]))
            })
            debut_col = fin_col
        matrices.insert(0, deepcopy(temp))
        debut_row = fin_row
    return matrices

