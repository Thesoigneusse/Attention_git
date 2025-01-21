import torch
from typing import List


def regrouper_indices_consecutifs(indices: List[int]) -> List[List[int]]:
    """
    Regroupe les indices consécutifs en sous-listes.
    Fonctionne sur une liste triée dans l'ordre décroissant.

    Exemple : [5, 4, 3, 1, 0] -> [[5, 4, 3], [1, 0]]

    >>> regrouper_indices_consecutifs([5, 4, 3, 1, 0])
    [[5, 4, 3], [1, 0]]
    >>> regrouper_indices_consecutifs([]) 
    """
    assert isinstance(indices, list), f"indices must be a List[int]. Current type: {type(indices)}"
    assert all(isinstance(i, int) for i in indices), f"indices must be a List[int]. Current type: {type(indices)}"
    groupes = []
    
    groupe = [indices[0]] if len(indices) >= 1 else None
    for i in range(1, len(indices)):
        if indices[i] == groupe[-1] - 1:  # Vérifie si l'index est consécutif au précédent
            groupe.append(indices[i])
        else:
            groupes.append(groupe)
            groupe = [indices[i]]
    groupes.append(groupe)
    
    # Permet de prendre en compte les lignes concernées par les BPEs mais qui n'en contiennent pas
    # i.e. les lignes dont l'indice précédent contient un marqueur BPE
    # for i in groupes: 
    #     i.append(i[-1]-1)
    return groupes if groupes[0] else None

def mean_matrices(matrices: List[torch.Tensor]) -> torch.Tensor:
    """Retourne la moyenne des matrices

    Args:
        matrices (List[torch.Tensor]): Liste de matrices

    Returns:
        torch.Tensor : la moyenne des matrices
    """
    return torch.stack(matrices).mean(dim=0)


if __name__ == "__main__":
    import doctest
    doctest.testmod()