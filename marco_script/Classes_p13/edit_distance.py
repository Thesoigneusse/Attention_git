
import sys
from typing import Dict, Optional, List
import torch

from Classes_p13.EditDistance import EditDistance
from Classes_p13.WordAlignement import WordAlignement

# Dictionnaire global de tokens pour optimiser le calcul des distances
__ed_dict__: Dict[str, int] = {}


def str_edit_distance(
    str_ref: str,
    str_hyp: str,
    model: Optional[object] = None,
    tokenizer: Optional[object] = None
) -> EditDistance:
    """
    Calcule la distance d’édition (edit distance) entre une phrase de référence et une phrase hypothèse.
    Renvoie un objet `EditDistance` contenant le nombre d’erreurs et les alignements.

    Args:
        str_ref (str): Phrase de référence (gold standard).
        str_hyp (str): Phrase hypothèse (prédiction).
        model (Optional[object]): Réservé pour compatibilité, non utilisé.
        tokenizer (Optional[object]): Réservé pour compatibilité, non utilisé.

    Returns:
        EditDistance: Objet contenant :
            - nombre_erreur_insertion
            - nombre_erreur_suppression
            - nombre_erreur_substitution
            - taille_tenseur_reference
            - alignements (liste de `WordAlignement`)
    """
    global __ed_dict__

    # Évite que le dictionnaire global ne devienne trop volumineux
    if len(__ed_dict__) > 100_000:
        __ed_dict__.clear()

    # Conversion en indices entiers
    ref_tokens: List[str] = str_ref.split()
    hyp_tokens: List[str] = str_hyp.split()

    for token in ref_tokens + hyp_tokens:
        __ed_dict__.setdefault(token, len(__ed_dict__))

    ref_tensor = torch.tensor([__ed_dict__[t] for t in ref_tokens], dtype=torch.long)
    hyp_tensor = torch.tensor([__ed_dict__[t] for t in hyp_tokens], dtype=torch.long)

    ins_weight = del_weight = sub_weight = 1.0

    x_size, y_size = ref_tensor.size(0), hyp_tensor.size(0)
    ed_matrix = torch.zeros((x_size + 1, y_size + 1), dtype=torch.float)

    # Initialisation des bords
    ed_matrix[1:, 0] = torch.arange(1, x_size + 1, dtype=torch.float)
    ed_matrix[0, 1:] = torch.arange(1, y_size + 1, dtype=torch.float)

    # Calcul de la matrice de distance
    for i in range(1, x_size + 1):
        for j in range(1, y_size + 1):
            cost = 0.0 if ref_tensor[i - 1] == hyp_tensor[j - 1] else sub_weight
            ed_matrix[i, j] = min(
                ed_matrix[i - 1, j] + del_weight,        # Suppression
                ed_matrix[i, j - 1] + ins_weight,        # Insertion
                ed_matrix[i - 1, j - 1] + cost           # Substitution ou match
            )

    # Backtracking
    n_ins = n_del = n_sub = 0
    alignements: List[WordAlignement] = []
    i, j = x_size, y_size

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ed_matrix[i, j] == ed_matrix[i - 1, j - 1] and ref_tensor[i - 1] == hyp_tensor[j - 1]:
            alignements.append(WordAlignement("match", i - 1, j - 1))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and ed_matrix[i, j] == ed_matrix[i - 1, j - 1] + sub_weight:
            alignements.append(WordAlignement("sub", i - 1, j - 1))
            n_sub += 1
            i, j = i - 1, j - 1
        elif i > 0 and ed_matrix[i, j] == ed_matrix[i - 1, j] + del_weight:
            alignements.append(WordAlignement("del", i - 1, None))
            n_del += 1
            i -= 1
        else:
            alignements.append(WordAlignement("ins", i if i > 0 else 0, j - 1))
            n_ins += 1
            j -= 1

    alignements.reverse()

    return EditDistance(
        nombre_erreur_insertion=n_ins,
        nombre_erreur_suppression=n_del,
        nombre_erreur_substitution=n_sub,
        taille_tenseur_reference=x_size,
        alignements=alignements
    )


def main(argv: List[str]) -> None:
    """
    Fonction principale de test : calcule la distance d’édition entre deux chaînes données en argument.

    Args:
        argv (List[str]): Liste des arguments de ligne de commande.
                          argv[1] = référence, argv[2] = hypothèse
    """
    if len(argv) < 3:
        print("Usage: python edit_distance.py <reference> <hypothese>")
        sys.exit(1)

    ref_str, hyp_str = argv[1], argv[2]

    print(f" * Computing edit distance between:\n * ref: {ref_str}\n * hyp: {hyp_str}\n ---")

    er_vals = str_edit_distance(ref_str, hyp_str)

    print(f" * ER: {er_vals.get_wer():.2f}")
    print(f" * Errors: ins={er_vals.nombre_erreur_insertion}, "
          f"del={er_vals.nombre_erreur_suppression}, sub={er_vals.nombre_erreur_substitution}\n ---")

    ref_tokens = ref_str.split()
    hyp_tokens = hyp_str.split()

    print(" * WordAlignement:")
    for a in er_vals.alignements:
        ref_tok = ref_tokens[a.index_reference] if a.index_reference < len(ref_tokens) else "-"
        hyp_tok = hyp_tokens[a.index_hypothese] if a.index_hypothese is not None and a.index_hypothese < len(hyp_tokens) else "-"
        print(f"   - {a.alignement_type}: ref='{ref_tok}' | hyp='{hyp_tok}'")

    print(" ---")


if __name__ == "__main__":
    main(sys.argv)
