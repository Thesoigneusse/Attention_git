import torch
from typing import List
from typing import Callable
import action_norm_tensor as ant
from Snt import Snt
import action_fusion_bpe as afb

class Matrice():
    def __init__(self, matrice: torch.Tensor) -> None:
        if not isinstance(matrice, torch.Tensor):
            matrice = torch.Tensor(matrice)
        self.matrice = matrice

    @property
    def matrice(self) -> torch.Tensor:
        return self._matrice
    @matrice.setter
    def matrice(self, matrice: torch.Tensor) -> None:
        self._matrice = matrice

    def __json__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__json__()

    # Padding Suppression
    @staticmethod
    def row_suppr_pad(matrice, row_list_suppr_pad: List[int]) -> torch.Tensor:
        """Supprime les lignes de la matrice à la position indiquée dans la liste_index

        Args:
            row_list_suppr_pad (List[int]): liste des positions des lignes à supprimer
        Exemples:
        >>> m1 = Matrice(torch.DoubleTensor([[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,24], [25,26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40,41,42], [43,44,45,46,47,48]]))
        >>> Matrice.row_suppr_pad(m1.matrice, [1,2,3,5])
        tensor([[ 1.,  2.,  3.,  4.,  5.,  6.],
                [25., 26., 27., 28., 29., 30.],
                [37., 38., 39., 40., 41., 42.],
                [43., 44., 45., 46., 47., 48.]])
        """
        assert isinstance(row_list_suppr_pad, list), f"row_list_suppr_pad doit être une liste. Current type: {type(row_list_suppr_pad)}"
        assert all(isinstance(index, int) for index in row_list_suppr_pad), f"row_list_suppr_pad doit être une liste d'entier. Current type: {[type(index) for index in row_list_suppr_pad]}"
        assert len(row_list_suppr_pad) == len(set(row_list_suppr_pad)), f"row_list_suppr_pad doit ne contenir qu'une fois chaque valeur. Current value: {row_list_suppr_pad}"
        masque = torch.ones(matrice.size(0), dtype=torch.bool)
        masque[row_list_suppr_pad] = False
        # Appliquer le masque pour supprimer les lignes
        return matrice[masque]

    def suppr_pad(self, row_list_suppr_pad: List[int]= None, col_list_suppr_pad: List[int]= None) -> None:
        """Supprime les lignes et les colonnes de la matrice à la position indiquée dans les listes_index

        Args:
            row_list_suppr_pad (List[int]): liste des positions des lignes à supprimer
            col_list_suppr_pad (List[int]): liste des positions des colonnes à supprimer
        Exemples:
        >>> m1 = Matrice(torch.DoubleTensor([[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,24], [25,26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40,41,42], [43,44,45,46,47,48]]))
        >>> m1.suppr_pad([1,2,3,5], [1,2,3,5])
        >>> print(m1.matrice)
        tensor([[ 1.,  5.],
                [25., 29.],
                [37., 41.],
                [43., 47.]])
        """
        if row_list_suppr_pad is not None:
            assert isinstance(row_list_suppr_pad, list), f"row_list_suppr_pad doit être une liste. Current type: {type(row_list_suppr_pad)}"
            assert len(row_list_suppr_pad) == len(set(row_list_suppr_pad)), f"row_list_suppr_pad doit ne contenir qu'une fois chaque valeur. Current value: {row_list_suppr_pad}"
            assert all(isinstance(index, int) for index in row_list_suppr_pad), f"row_list_suppr_pad doit être une liste d'entier. Current type: {[type(index) for index in row_list_suppr_pad]}"
            self.matrice = self.row_suppr_pad(self.matrice, row_list_suppr_pad)
        if col_list_suppr_pad is not None:
            assert isinstance(col_list_suppr_pad, list), f"col_list_suppr_pad doit être une liste. Current type: {type(col_list_suppr_pad)}"
            assert all(isinstance(index, int) for index in col_list_suppr_pad), f"col_list_suppr_pad doit être une liste d'entier. Current type: {[type(index) for index in col_list_suppr_pad]}"
            assert len(col_list_suppr_pad) == len(set(col_list_suppr_pad)), f"row_list_suppr_pad doit ne contenir qu'une fois chaque valeur. Current value: {col_list_suppr_pad}"
            self.matrice = self.row_suppr_pad(self.matrice.transpose(1,0), col_list_suppr_pad).transpose(1,0)

    # BPE fusion
    @staticmethod
    def action_fusion_bpe(rows: torch.Tensor, action: str = "max") -> torch.Tensor:
        """
        Fusionne plusieurs lignes d'un tensor 2D avec un tensor 1D.

        Args:
            rows: Tensor 2D des lignes à fusionner
            action: Fonction de fusion (par défaut, torch.max)
        Returns:
            torch.Tensor: Résultat de l'action appliquée à toutes les lignes
        TODO: faire un doctest
        """
        result = None
        if action == "max":
            result = afb.max(rows)
        elif action == "mean":
            result = afb.mean(rows)
        assert result is not None, f"Problème action_fusion_bpe. action vs. result: {action} vs. {result}"
        return result

    @staticmethod
    def row_fusion_bpe(matrice: torch.Tensor, row_list_fusion_bpe: List[int] = None, action: str = "max") -> torch.Tensor:
        """
        Fusionne les lignes spécifiées dans row_list_fusion_bpe.

        Args:
            matrice: Tensor (matrice)
            row_list_fusion_bpe: Liste de liste des indices de lignes à fusionner
            action: Fonction de fusion
        Returns:
            torch.Tensor: Matrice avec les lignes fusionnées
        TODO: faire un doctest

        """
        # row_list_fusion_bpe.sort(reverse=True)  # Trier les indices en ordre décroissant

        # Regrouper les indices consécutifs en groupes
        groupes = row_list_fusion_bpe
        for groupe in groupes:  # Traiter les groupes en ordre décroissant
            ligne_fusionnee = Matrice.action_fusion_bpe(rows = matrice[groupe], action=action)
            matrice[groupe[-1]] = ligne_fusionnee  # Remplacer la première ligne du groupe
            matrice = torch.cat((matrice[:groupe[-1] + 1], matrice[groupe[0] + 1:]))  # Supprimer les autres lignes
        return matrice

    def fusion_bpe(self, row_list_fusion_bpe: List[int]= None, col_list_fusion_bpe: List[int]= None, action: str = "max") -> None:
        """
        Fusionne les lignes et colonnes spécifiées dans les listes row_list_fusion_bpe et col_list_fusion_bpe.

        Args:
            row_list_fusion_bpe: Liste des indices de lignes à fusionner
            col_list_fusion_bpe: Liste des indices de colonnes à fusionner
            action: Fonction de fusion
        TODO: faire un doctest
        """
        # Fusion des lignes
        if row_list_fusion_bpe is not None and len(row_list_fusion_bpe)>= 1:
            self.matrice = self.row_fusion_bpe(self.matrice, row_list_fusion_bpe, action=action)

        # Fusion des colonnes (en transposant la matrice)
        if col_list_fusion_bpe is not None and len(col_list_fusion_bpe) >= 1:
            self.matrice = self.row_fusion_bpe(self.matrice.transpose(1, 0), col_list_fusion_bpe, action=action).transpose(1, 0)


    # norm tensor
    def action_norm_tensor(row: torch.Tensor, medium: str = "minmax" ) -> torch.Tensor:
        """
        Applique une action sur une ligne et renvoie le résultat

        Args:
            row: Ligne à traiter
            medium: action à effectuer
        Returns:
            torch.Tensor: Résultat de l'action appliquée à la ligne

        Exemples:
        >>> Matrice.action_norm_tensor(torch.DoubleTensor([0,2,3,4,5,6]), medium = "minmax")
        tensor([0.0000, 0.0000, 0.2500, 0.5000, 0.7500, 1.0000])
        """
        if medium =="minmax":
            return ant.norm_by_min_max(row)
        elif medium == "max":
            return ant.norm_by_max(row)
        else:
            raise ValueError(f"medium doit être 'minmax' ou'max'. Current value: {medium}")

    def norm_tensor(self, precision: int = 3, medium="minmax"): 
        """Normalise les poids de la matrice

        Args:
            precision (int, optional): valeur pour l'arrondi. Defaults to 2.
            medium (str, optional): choix pour l'action à effectuer sur la ligne. Defaults to "minmax".
        
        Exemples:
        >>> m1 = Matrice(torch.DoubleTensor([[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,24], [25,26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40,41,42], [43,44,45,46,47,48]]))
        >>> m1.norm_tensor(precision=2, medium='minmax')
        >>> print(m1.matrice)
        tensor([[0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000]])
        >>> m2 = Matrice(torch.DoubleTensor([[1,2,3,4,5,6], [7,8,9,10,11,12], [13,14,15,16,17,18], [19,20,21,22,23,24], [25,26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40,41,42], [43,44,45,46,47,48]]))
        >>> m2.norm_tensor(precision=2, medium='max')
        >>> print(m2.matrice)
        tensor([[0.1700, 0.3300, 0.5000, 0.6700, 0.8300, 1.0000],
                [0.5800, 0.6700, 0.7500, 0.8300, 0.9200, 1.0000],
                [0.7200, 0.7800, 0.8300, 0.8900, 0.9400, 1.0000],
                [0.7900, 0.8300, 0.8800, 0.9200, 0.9600, 1.0000],
                [0.8300, 0.8700, 0.9000, 0.9300, 0.9700, 1.0000],
                [0.8600, 0.8900, 0.9200, 0.9400, 0.9700, 1.0000],
                [0.8800, 0.9000, 0.9300, 0.9500, 0.9800, 1.0000],
                [0.9000, 0.9200, 0.9400, 0.9600, 0.9800, 1.0000]])
        >>> m3 = Matrice(torch.DoubleTensor([[0,2,3,4,5,6], [7,8,9,0,11,12], [13,14,15,16,17,18], [19,20,0,0,23,24], [25,26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40,41,42], [43,44,45,46,47,48]]))
        >>> m3.norm_tensor(precision=2, medium='minmax')
        >>> print(m3.matrice)
        tensor([[0.0000, 0.0000, 0.2500, 0.5000, 0.7500, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.0000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.0000, 0.0000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000],
                [0.0000, 0.2000, 0.4000, 0.6000, 0.8000, 1.0000]])
        """
#         print(f"[DEBUG]self.matrice: {self.matrice}")
        for itok in range(self.matrice.size(dim=0)):
            mask_not_nul = self.matrice[itok] != 0
            if len(self.matrice[itok]) != 1 \
                and not torch.numel(self.matrice[itok][mask_not_nul]) == 0 \
                and not torch.all(self.matrice[itok][mask_not_nul] == self.matrice[itok][mask_not_nul][0]):
                # Exclu les cas causant un NaN à cause d'une normalisation via le minimum,
                # C'est-à-dire le cas d'une valeur unique dans le vecteur
                # Ainsi que le cas où toutes les valeurs non nulles sont égales
                self.matrice[itok] = Matrice.action_norm_tensor(self.matrice[itok], medium = medium)
            # import pudb; pudb.set_trace()

            # Issue with torch.round() : not accurate enough
            # print(f"avant : {self.matrice[itok]}")
            # self.matrice = self.matrice.round(decimals = precision)
            # print(f"après : {self.matrice[itok]}")

    # suppr_inf_uniform
    def suppr_inf(self, medium: str= "inf_uniform", value = None):
        size = self.matrice.size()
        if size[0] > 1 :
            if medium =="inf_uniform" :
                self.matrice[self.matrice < (1/size[1])] = 0
            elif medium == "inf_uniform_modif":
                self.matrice[self.matrice < (1/size[1])] = 1/size[1]
            elif medium == "value":
                    assert value is not None and isinstance(value, int), f"avec l'option value, clean_matrice doit recevoir une valeur"
                    self.matrice[self.matrice < 1/value] = 0

    def ecriture_xslx(self, crt: Snt, ctx: Snt, absolute_folder, filename, precision = 2, create_folder_path = False):
        """Écrit la matrice au format xslx

        Args:
            matrice (torch.DoubleTensor): matrice des poids d'attentions
            crt (Snt): phrase courante
            ctx (Snt): phrase de contexte
            absolute_folder (str): chemin absolu vers le dossier de sauvegarde
            filename (str): nom du fichier de sauvegarde
            precision (int, optional): nombre de chiffres significatifs à garder. Defaults to .
            create_folder_path (bool, optional): indique s'il faut créer l'arborescence ou non. Defaults to False.
        """
        """
        TODO: Corriger la façon de faire avec l'écriture de la précision. Trouver une alternative à torch.round() 
        pour ne garder que les deniers chiffres significatifs voulus.
        """
        # Check the path
        import Utils_data
        Utils_data.check_path(absolute_folder=absolute_folder, create_folder_path=create_folder_path)
        import xlsxwriter
        workbook = xlsxwriter.Workbook(f"{absolute_folder}/{filename}.xlsx")
        worksheet = workbook.add_worksheet()

        # Définition un format pour mettre en évidence les valeurs maximales
        highlight_format = workbook.add_format({'bg_color': 'cyan', 'bold': True})

        # Ecritures des phrases respectivement courante et de contexte
        worksheet.write(0,0, f"{crt.identifiant}-k{str(int(crt.identifiant) - int(ctx.identifiant))}")
        for row_idx, tok in enumerate(crt.tokens, start=1):
            worksheet.write(row_idx, 0, tok)
        for col_idx, tok in enumerate(ctx.tokens, start=1):
            worksheet.write(0, col_idx, tok)

        for row_idx in range(self.matrice.shape[0]):
            max_value = torch.max(self.matrice[row_idx])
            for col_idx in range(self.matrice.shape[1]):
                value = self.matrice[row_idx, col_idx].item()
                if value == max_value:
                    worksheet.write(row_idx + 1, col_idx + 1, str(value)[:2+precision], highlight_format)
                elif value == 0.0:
                    worksheet.write(row_idx + 1, col_idx + 1, ".")
                else:
                    worksheet.write(row_idx + 1, col_idx + 1, str(value)[:2+precision])
        workbook.close()




if __name__ == '__main__':
    import doctest
    doctest.testmod()
    print(f"[DEBUG] Doctest clear")
    
    # print()
    # print(f"[DEBUG] matrice.norm()")  
    # m1 = Matrice(torch.DoubleTensor([[0,2,3,4,5,6], [7,8,8,0,11,12], [13,14,15,16,17,18], [19,20,0,0,23,24], [25,26,27,28,29,30], [31,32,33,34,35,36], [37,38,39,40,41,42], [43,44,45,46,47,48]]))
    # m1.norm_tensor()
    # print(m1.matrice)
    # m1.ecriture_xslx(crt=Snt.Snt(identifiant=1, tokens=["1", "2", "3", "4", "5", "6", "7", "8"]),
    #                 ctx=Snt.Snt(identifiant=0, tokens=["1", "2", "3", "4", "5", "6"]),
    #                 absolute_folder=path,
    #                 filename="matrice_test",
    #                 create_folder_path=False)

    
    


