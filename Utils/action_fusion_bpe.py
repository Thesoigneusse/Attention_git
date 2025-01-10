import torch
def max(row: torch.Tensor) -> torch.Tensor:
    """
    Retourne le maximum d'une matrice torch.
    """
    return torch.max(row, dim=0).values

def mean(row: torch.Tensor) -> torch.Tensor:
    """
    Retourne la moyenne d'une matrice torch.
    """
    return torch.mean(row, dim=0)

if __name__ == '__main__':
    import doctest  
    doctest.testmod()

    t1 = torch.Tensor([[1,2,3,4,5], [6,7,8,9,10], [11,12,13,14,15]])
    
    print(mean(t1))
