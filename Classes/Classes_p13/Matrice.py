from __future__ import annotations
import torch
from typing import Iterable, Union, List, Dict, Any, TypeVar, Optional

T = TypeVar('T', bound="Matrice")
class Matrice:
    """
    Classe représentant une matrice encapsulant un torch.Tensor.

    Parameters
    ----------
    data : torch.Tensor | Iterable[Iterable[float]]
        Les données de la matrice. Si un iterable d'iterables est fourni,
        il sera converti en torch.Tensor.

    Raises
    ------
    TypeError
        Si `data` n'est ni un torch.Tensor ni un iterable bidimensionnel.
    ValueError
        Si la matrice fournie n'est pas strictement bidimensionnelle (2D).
    """

    def __init__(self, data: Union[torch.Tensor, Iterable[Iterable[float]]]):
        # Conversion éventuelle vers Tensor
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            try:
                tensor = torch.tensor(list(list(row) for row in data), dtype=torch.float32)
            except Exception as e:
                raise TypeError("Impossible de convertir `data` en torch.Tensor 2D.") from e

        if tensor.ndim != 2:
            raise ValueError(f"La matrice doit être 2D, mais ndim={tensor.ndim}.")

        self.data: torch.Tensor = tensor

    def __repr__(self) -> str:
        return f"Matrice(shape={tuple(self.data.shape)}, data=\n{self.data})"

    # ------------------------------------------------------------------
    # Propriétés utiles
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, int]:
        """Retourne la forme (lignes, colonnes) de la matrice."""
        rows, cols = self.data.shape
        return (rows, cols)

    def size(self, dim: Optional[int] = None) -> int | torch.Size:
        """Retourne la taille de la matrice le long d'une dimension donnée ou la forme complète.

        Parameters
        ----------
        dim : Optional[int], optional
            La dimension pour laquelle retourner la taille. Si None, retourne la forme complète.

        Returns
        -------
        int | torch.Size
            La taille le long de la dimension spécifiée ou la forme complète de la matrice.
        """
        return self.data.size(dim) if dim is not None else self.data.size()

    # ------------------------------------------------------------------
    # Opérations basiques
    # ------------------------------------------------------------------

    def transpose(self) -> Matrice:
        """Retourne la transposée de la matrice."""
        return Matrice(self.data.T)

    def __add__(self, other: Matrice) -> Matrice:
        """
        Addition matricielle.

        Parameters
        ----------
        other : Matrice
            La matrice ou le scalaire à ajouter.

        Raises
        ------
        TypeError
            Si `other` n'est pas de type Matrice.

        Returns
        -------
        Matrice
            Une nouvelle matrice résultant de l'addition.
        """
        if not isinstance(other, Matrice):
            raise TypeError("L'addition n'est définie qu'entre objets Matrice.")
        return Matrice(self.data + other.data)

    def __matmul__(self, other: Matrice) -> Matrice:
        """
        Produit matriciel (opérateur @).

        Parameters
        ----------
        other : Matrice

        Raises
        ------
        TypeError
            Si `other` n'est pas une Matrice.
        RuntimeError
            Si les dimensions ne sont pas compatibles pour le produit matriciel.

        Returns
        -------
        Matrice
            Le résultat du produit.
        """
        if not isinstance(other, Matrice):
            raise TypeError("Le produit matriciel n'est défini qu'entre objets Matrice.")
        return Matrice(self.data @ other.data)

    def elementwise_mul(self, other: Matrice) -> Matrice:
        """
        Produit élément par élément (Hadamard).

        Parameters
        ----------
        other : Matrice

        Returns
        -------
        Matrice
        """
        if not isinstance(other, Matrice):
            raise TypeError("Le produit élémentaire n'est défini qu'entre objets Matrice.")
        return Matrice(self.data * other.data)

    def round(self, precision: int = 2) -> "Matrice":
        """
        Retourne une nouvelle Matrice dont les valeurs sont arrondies
        à la précision spécifiée, en utilisant torch.round.

        Parameters
        ----------
        precision : int, optional
            Nombre de décimales à conserver (par défaut 2).

        Returns
        -------
        Matrice
            Une nouvelle matrice arrondie.
        """
        scale = 10 ** precision
        rounded_tensor = torch.round(self.data * scale) / scale
        return Matrice(rounded_tensor)

    def tojson(self, precision: int = 2) -> Dict[str, Any]:
        """
        Convertit la matrice en un dictionnaire JSON-compatible,
        arrondi avec torch.round via self.round().

        Parameters
        ----------
        precision : int, optional
            Nombre de décimales à conserver (par défaut 2).

        Returns
        -------
        Dict[str, Any]
            Dictionnaire contenant shape, dtype, precision et data.
        """

        rounded = self.round(precision)

        return {
            "shape": list(rounded.data.size()),
            "dtype": str(rounded.data.dtype),
            "precision": precision,
            "data": rounded.data.tolist()
        }

    @staticmethod
    def fromjson(obj: Dict[str, Any]) -> "Matrice":
        """
        Reconstruit une matrice à partir du dictionnaire produit par tojson().

        Parameters
        ----------
        obj : Dict[str, Any]
            Le dictionnaire contenant 'data', 'dtype', 'shape'.

        Returns
        -------
        Matrice
            La matrice reconstruite.

        Raises
        ------
        KeyError
            Si les clés nécessaires sont manquantes.
        TypeError
            Si les données ne sont pas dans un format attendu.
        ValueError
            Si la matrice n'est pas 2D.
        """

        if not all(k in obj for k in ("data", "dtype", "shape")):
            raise KeyError("Le dictionnaire doit contenir 'data', 'dtype' et 'shape'.")

        data = obj["data"]
        dtype = obj["dtype"]

        # Convertit "torch.float32" → torch.float32
        try:
            torch_dtype = getattr(torch, dtype.split(".")[1])
        except Exception:
            raise TypeError(f"Le dtype '{dtype}' est invalide pour torch.")

        # Reconstruction du tensor
        try:
            tensor = torch.tensor(data, dtype=torch_dtype)
        except Exception as e:
            raise TypeError("Impossible de reconstruire le torch.Tensor depuis les données JSON.") from e

        if tensor.dim() != 2:
            raise ValueError("La matrice reconstruite doit être 2D.")

        if not tensor.is_floating_point():
            raise TypeError("Le dtype doit être un type flottant.")

        return Matrice(tensor)
    
    def suppr_ligne_i(self: T, index: int) -> T:
        """Retourne une nouvelle matrice sans la ligne d'indice `index`.

        Args:
            index (int): indice de la ligne à supprimer.

        Raises:
            IndexError: si l'indice est hors bornes.

        Returns:
            Matrice: nouvelle instance sans la ligne spécifiée.
        """
        if index < 0 or index >= self.data.shape[0]:
            raise IndexError("Index de ligne hors bornes.")

        new_tensor = torch.cat([
            self.data[:index, ...],
            self.data[index+1:, ...]
        ], dim=0)

        return type(self)(new_tensor)

    def suppr_lignes_from_i1_to_i2(self: T, index1: int, index2: int) -> T:
        """Retourne une nouvelle matrice sans les lignes d'indices compris
        entre index1 (inclus) et index2 (exclu).

        Args:
            index1 (int): début de la plage.
            index2 (int): fin de la plage (exclu).

        Raises:
            IndexError: si la plage dépasse les dimensions.

        Returns:
            Matrice: nouvelle matrice sans les lignes sélectionnées.
        """
        if index1 < 0 or index2 > self.data.shape[0] or index1 >= index2:
            raise IndexError("Plage de suppression invalide.")

        new_tensor = torch.cat([
            self.data[:index1, ...],
            self.data[index2:, ...]
        ], dim=0)

        return type(self)(new_tensor)

    # ------------------------------------------------------------------
    # Outils de création
    # ------------------------------------------------------------------

    @staticmethod
    def zeros(rows: int, cols: int) -> Matrice:
        """Crée une matrice de zéros."""
        return Matrice(torch.zeros((rows, cols)))

    @staticmethod
    def ones(rows: int, cols: int) -> Matrice:
        """Crée une matrice de uns."""
        return Matrice(torch.ones((rows, cols)))

    @staticmethod
    def identity(n: int) -> Matrice:
        """Crée une matrice identité n×n."""
        return Matrice(torch.eye(n))

    # ------------------------------------------------------------------
    # Opérations Complexes : BPEs
    # ------------------------------------------------------------------
    def fusion_groupe(self, 
                      index_list: List[int], 
                      medium: str = "max") -> "Matrice":
        """
        Fusionne plusieurs lignes de la matrice selon la méthode spécifiée.

        Args:
            index_list (List[int]): indices des lignes à fusionner
            medium (str, optional): 'max' ou 'mean'. Defaults to 'max'.

        Raises:
            NotImplementedError: si medium inconnu

        Returns:
            Matrice: nouvelle matrice avec les lignes fusionnées

        Example:
        >>> Matrice(torch.tensor([[1.,2.,3.], [3.,2.,1.]])).fusion_ligne([1,0], medium='mean')
        Matrice([[2., 2., 2.]])
        """
        dict_action = {
            "max": lambda t: torch.max(t, dim=0).values,
            "mean": lambda t: torch.mean(t, dim=0)
        }

        if medium not in dict_action:
            raise NotImplementedError(
                f"fusion_ligne: unknown medium.\nSupported: {list(dict_action.keys())}\nGot: {medium}"
            )

        # Trier les indices pour éviter des problèmes lors de la suppression
        sorted_indices = sorted(index_list, reverse=True)
        tensor_to_fuse = self.data[sorted_indices, ...]
        fused_line = dict_action[medium](tensor_to_fuse)

        # Créer un nouveau tensor
        new_tensor = self.data.clone()
        new_tensor[sorted_indices[0], :] = fused_line
        # Supprimer les lignes restantes
        for idx in reversed(sorted_indices[1:]):
            new_tensor = torch.cat([new_tensor[:idx, ...], new_tensor[idx+1:, ...]], dim=0)

        return Matrice(new_tensor)

    def fusion_bpe(self, 
        row_list_groupes: List[List[int]] | None = None, 
        col_list_groupes: List[List[int]] | None = None, 
        medium: str = 'max'
    ) -> "Matrice":
        """
        Fusionne toutes les lignes/colonnes spécifiées selon medium.

        Args:
            row_list_groupes (List[List[int]], optional): groupes de lignes à fusionner
            col_list_groupes (List[List[int]], optional): groupes de colonnes à fusionner
            medium (str, optional): 'max' ou 'mean'

        Returns:
            Matrice: nouvelle matrice fusionnée

        Example:
        >>> Matrice(torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.],[13.,14.,15.]])).fusion_bpe(row_list_groupes=[[4,3],[2,1]])
        Matrice([[ 1.,  2.,  3.],
                 [ 7.,  8.,  9.],
                 [13., 14., 15.]])
        """
        result = self
        if row_list_groupes is not None:
            for groupe in row_list_groupes:
                result = result.fusion_groupe(groupe, medium=medium)
        if col_list_groupes is not None:
            for groupe in col_list_groupes:
                result = result.transpose().fusion_groupe(groupe, medium=medium).transpose()
        return result

    # ------------------------------------------------------------------
    # Opérations Complexes : Normalisation
    # ------------------------------------------------------------------
    def norm_tenseur(self, medium: str = "minmax") -> "Matrice":
        """
        Normalise le tenseur selon la méthode spécifiée.

        Cette fonction applique une normalisation **ligne par ligne** sur la matrice.

        Parameters
        ----------
        medium : str, optional
            Méthode de normalisation à appliquer. Les options sont :
            - "minmax" : normalise chaque ligne entre 0 et 1 selon la formule 
                        (x - min) / (max - min)
            - "max" : divise chaque élément d'une ligne par le maximum de la ligne

        Returns
        -------
        Matrice
            Une nouvelle instance de `Matrice` contenant les valeurs normalisées.

        Raises
        ------
        NotImplementedError
            Si `medium` n'est pas reconnu.

        Examples
        --------
        >>> M = Matrice(torch.tensor([[1.,2.,3.],[4.,5.,6.]]))
        >>> M.norm_tenseur("minmax").data
        tensor([[0., 0.5, 1.],
                [0., 0.5, 1.]])
        >>> M.norm_tenseur("max").data
        tensor([[0.3333, 0.6667, 1.0000],
                [0.6667, 0.8333, 1.0000]])
        """

        if medium == "minmax":
            min_val = self.data.min(dim=1, keepdim=True).values
            max_val = self.data.max(dim=1, keepdim=True).values
            new_data = (self.data - min_val) / (max_val - min_val + 1e-8)
        elif medium == "max":
            row_max = self.data.max(dim=1, keepdim=True).values
            new_data = self.data / (row_max + 1e-8)
        else:
            raise NotImplementedError(f"norm_tenseur: unknown medium '{medium}'")
        return Matrice(new_data)

    # ------------------------------------------------------------------
    # Opérations Complexes : Suppression de valeurs inférieures à un seuil
    # ------------------------------------------------------------------
    def suppr_inf(self, medium: str = "suppr_inf_uniform", value: Optional[int] = None) -> "Matrice":
        """
        Supprime ou modifie les valeurs inférieures à un seuil par ligne.

        Cette fonction peut supprimer ou remplacer les petites valeurs selon trois méthodes.

        Parameters
        ----------
        medium : str, optional
            Méthode à appliquer. Options :
            - "suppr_inf_uniform" : remplace les valeurs inférieures à 1/n_colonnes par 0
            - "to_uniform" : remplace les valeurs inférieures à 1/n_colonnes par 1/n_colonnes
            - "suppr_inf_value" : remplace les valeurs inférieures à 1/value par 0
        value : int, optional
            Valeur entière nécessaire si `medium` == "suppr_inf_value". Défaut None.

        Returns
        -------
        Matrice
            Une nouvelle instance de `Matrice` après suppression/modification des valeurs.

        Raises
        ------
        ValueError
            Si `medium` == "suppr_inf_value" et que `value` n'est pas fourni ou n'est pas un entier.
        NotImplementedError
            Si `medium` n'est pas reconnu.

        Examples
        --------
        >>> M = Matrice(torch.tensor([[0.1,0.2,0.7],[0.4,0.4,0.2]]))
        >>> M.suppr_inf().data
        tensor([[0., 0., 0.7],
                [0.4, 0.4, 0.]])
        >>> M.suppr_inf(medium="to_uniform").data
        tensor([[0.3333, 0.3333, 0.7],
                [0.4, 0.4, 0.3333]])
        """


        new_data = self.data.clone()
        n_rows, n_cols = new_data.shape
        if n_rows > 1:
            if medium == "suppr_inf_uniform":
                threshold = 1 / n_cols
                new_data[new_data < threshold] = 0
            elif medium == "to_uniform":
                threshold = 1 / n_cols
                new_data[new_data < threshold] = threshold
            elif medium == "suppr_inf_value":
                if value is None or not isinstance(value, int):
                    raise ValueError("suppr_inf_value requires an integer 'value'")
                new_data[new_data < 1/value] = 0
            else:
                raise NotImplementedError(f"suppr_inf unknown medium '{medium}'")
        return Matrice(new_data)

    # ------------------------------------------------------------------
    # Opérations Complexes : Suppression des lignes ou colonnes de padding
    # ------------------------------------------------------------------
    def suppr_pad(
        self, 
        row_list_suppr_pad: Optional[List[int]] = None, 
        col_list_suppr_pad: Optional[List[int]] = None
    ) -> "Matrice":
        """
        Supprime des lignes et colonnes correspondant aux indices de padding.

        Parameters
        ----------
        row_list_suppr_pad : List[int], optional
            Liste des indices de lignes à supprimer. Les indices doivent être valides et
            la liste est traitée en ordre décroissant.
        col_list_suppr_pad : List[int], optional
            Liste des indices de colonnes à supprimer. Les indices doivent être valides
            et la liste est traitée en ordre décroissant.

        Returns
        -------
        Matrice
            Nouvelle instance de `Matrice` après suppression des lignes et colonnes spécifiées.

        Raises
        ------
        AssertionError
            Si l’un des indices n’est pas un entier ou si les listes ne sont pas correctes.

        Examples
        --------
        >>> M = Matrice(torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]]))
        >>> M.suppr_pad(row_list_suppr_pad=[0], col_list_suppr_pad=[2]).data
        tensor([[5.],
                [8.]])
        """

        new_data = self.data.clone()
        # Lignes
        if row_list_suppr_pad:
            rows_to_keep = [i for i in range(new_data.shape[0]) if i not in row_list_suppr_pad]
            new_data = new_data[rows_to_keep, :]
        # Colonnes
        if col_list_suppr_pad:
            cols_to_keep = [i for i in range(new_data.shape[1]) if i not in col_list_suppr_pad]
            new_data = new_data[:, cols_to_keep]
        return Matrice(new_data)


if __name__ == "__main__":
    data = torch.tensor([
        [0.1, 0.2, 0.7],
        [0.4, 0.4, 0.2],
        [0.13, 0.27, 0.6],
        [0.33, 0.33, 0.34],
        [0.30, 0.35, 0.35]
    ], dtype=torch.float32)

    M = Matrice(data)

    print(M)