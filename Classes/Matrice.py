import torch
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Classes.Snt import Snt

class TorchMatrice(torch.Tensor):
    def __new__(cls, data, *args, **kwargs):
        instance = super().__new__(cls, data, *args, **kwargs)
        return instance

    def __init__(self, data):
        pass

    def suppr_ligne_i(tenseur, index):
        return torch.cat([tenseur[:index,...], tenseur[index+1:, ...]])

    def suppr_lignes_from_i1_to_i2(self, index1, index2):
        return torch.cat([self[:index1,...], self[index2:, ...]])
        
    def fusion_ligne(self, index, medium="max"):
        """Retourne la fusion entre plusieurs lignes de self.data

        Args:
            index1 (List[int]): Liste des index des lignes à fusionner.
            medium (str, optional): méthode de fusion. Defaults to "max".

        Raises:
            NotImplementedError: Erreur survenant si le medium n'est pas compris dans le dictionnaire d'action.

        Returns:
            torch.Tensor: torch.Tensor de sortie correspondant à la fusion des deux lignes du torch.Tensor d'entrée

        Tests:
        >>> TorchMatrice([[1,2,3], [3,2,1]]).fusion_ligne([1,0], medium='mean')
        TorchMatrice([[2., 2., 2.]])
        >>> TorchMatrice([[1,2,3], [3,2,1]]).fusion_ligne([1,0], medium='max')
        TorchMatrice([[3., 2., 3.]])
        """
        dict_action = {"max": lambda t: torch.max(t, dim = 0).values,
                        "mean": lambda t: torch.mean(t , dim =0)}
        if medium in dict_action:
            self[index, ...] = dict_action[medium](self[index])
            self = self.suppr_lignes_from_i1_to_i2(index[-1], index[0])
            # self = torch.cat([self[:index[-1] , ...],  self[index[0]:, ...]])
        else:
            raise NotImplementedError(f"fusion_ligne: unknown medium.\nSupported medium : {dict_action.keys()}\nCurrent medium: '{medium}'")
        return self

    def fusion_bpe(self, groupes, medium='max'):
        """Fusionne tout les BPEs présent dans groupes.

        Args:
            groupes (List[List[int]]): Liste décroissante de liste décroissante d'index
            medium (str, optional): méthode de fusion. Defaults to 'max'.

        Tests:
        >>> TorchMatrice([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]).fusion_bpe(groupes= [[4,3], [2,1]])
        TorchMatrice([[ 1.,  2.,  3.],
                      [ 7.,  8.,  9.],
                      [13., 14., 15.]])
        """
        for groupe in groupes:
            self = self.fusion_ligne(groupe, medium=medium)
        return self

    def norm_tenseur(self, medium="minmax"):
        """Normalise le tenseur selon l'action 'medium'

        Args:
            medium (str, optional): Action de normalisation à appliquer sur le tenseur. Defaults to "minmax".

        Tests:
        >>> TorchMatrice([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]).norm_tenseur()
        TorchMatrice([[0.0000, 0.5000, 1.0000],
                      [0.0000, 0.5000, 1.0000],
                      [0.0000, 0.5000, 1.0000],
                      [0.0000, 0.5000, 1.0000],
                      [0.0000, 0.5000, 1.0000]])
        >>> TorchMatrice([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]).norm_tenseur(medium = "max")
        TorchMatrice([[0.3333, 0.6667, 1.0000],
                      [0.6667, 0.8333, 1.0000],
                      [0.7778, 0.8889, 1.0000],
                      [0.8333, 0.9167, 1.0000],
                      [0.8667, 0.9333, 1.0000]])
        """
        from Utils import action_norm_tensor as ant
        dict_action = {"minmax": ant.norm_by_min_max,
                        "max": ant.norm_by_max}
        for i in range(self.size(dim=0)):
            self[i] = dict_action[medium](self[i])
        return self

    def suppr_inf(self, medium: str ="suppr_inf_uniform", value: int = None):
        """Supprime les valeurs de chaque ligne du tensor en dessous de l'uniforme.
            Choices: 
            - suppr_inf_uniform : supprime les valeurs inférieures à la distribution uniforme de la ligne.
            - suppr_inf_value : supprime les valeurs inférieures à une valeur donnée.
            - to_uniform : Transforme les valeurs inférieures à la distribution uniforme en vvaleurs égales à la distribution uniforme

        Args:
            medium (str, optional): méthode à appliquer. Defaults to "suppr_inf_uniform".
            value (_type_, optional): Dans le cas d'une suppression uniforme par rapport à une valeur, indique la valeur. Defaults to None.

        Raises:
            NotImplementedError: Retourne une erreur dans le cas où la méthode à utiliser n'est pas implémentée.

        Tests:
        >>> t1 = TorchMatrice([[0.1, 0.2, 0.7], [0.4,0.4,0.2], [0.13,0.27,0.6], [0.33,0.33,0.34], [0.30,0.35,0.35]])
        >>> t1.suppr_inf()
        >>> print(t1)
        TorchMatrice([[0.0000, 0.0000, 0.7000],
                      [0.4000, 0.4000, 0.0000],
                      [0.0000, 0.0000, 0.6000],
                      [0.0000, 0.0000, 0.3400],
                      [0.0000, 0.3500, 0.3500]])
        >>> t1 = TorchMatrice([[0.1, 0.2, 0.7], [0.4,0.4,0.2], [0.13,0.27,0.6], [0.33,0.33,0.34], [0.30,0.35,0.35]])
        >>> t1.suppr_inf(medium = 'to_uniform')
        >>> print(t1)
        TorchMatrice([[0.3333, 0.3333, 0.7000],
                      [0.4000, 0.4000, 0.3333],
                      [0.3333, 0.3333, 0.6000],
                      [0.3333, 0.3333, 0.3400],
                      [0.3333, 0.3500, 0.3500]])
        """
        size = self.size()
        if size[0] > 1 :
            if medium =="suppr_inf_uniform" :
                self[self < (1/size[1])] = 0
            elif medium == "to_uniform":
                self[self < (1/size[1])] = 1/size[1]
            elif medium == "suppr_inf_value":
                assert value is not None and isinstance(value, int), f"avec l'option value, clean_matrice doit recevoir une valeur"
                self[self < 1/value] = 0
            else:
                raise NotImplementedError(f"suppr_inf ne supporte pas la suppression uniforme avec la méthode: '{medium}'")

    def ecriture_xslt(self, crt: 'Snt', ctx: 'Snt', absolute_folder: str, filename: str, precision: int = 2, create_folder_path: bool = False) -> None:
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
        assert len(crt) == self.size(dim = 0), f"[DEBUG]Phrase {crt.identifiant}. Les lignes de la matrice doivent correspondre au nombre de mots dans la phrase courante {crt.identifiant}. l_matrice vs. nb_tokens: {self.size(dim = 0)} vs. {len(crt)}"
        assert len(ctx) == self.size(dim = 1), f"[DEBUG]Phrase {crt.identifiant}. Les colonnes de la matrice doivent correspondre au nombre de mots dans la phrase de contexte {ctx.identifiant}. l_matrice vs. nb_tokens: {self.size(dim = 1)} vs. {len(ctx)}"
        from Utils import Utils_data
        Utils_data.check_path(absolute_folder=absolute_folder, create_folder_path=create_folder_path)
        import xlsxwriter
        workbook = xlsxwriter.Workbook(f"{absolute_folder}/{filename}.xlsx")
        worksheet = workbook.add_worksheet()

        # Définition un format pour mettre en évidence les valeurs maximales
        highlight_format = workbook.add_format({'bg_color': 'cyan', 'bold': True})

        # Ecritures des phrases respectivement courante et de contexte
        # worksheet.write(0,0, f"{crt.identifiant}-k{str(int(crt.identifiant) - int(ctx.identifiant))}")
        worksheet.write(0,0, f"{crt.identifiant}-{str(int(crt.identifiant) - int(ctx.identifiant))}")
        for row_idx, tok in enumerate(crt.tokens, start=1):
            worksheet.write(row_idx, 0, tok)
        for col_idx, tok in enumerate(ctx.tokens, start=1):
            worksheet.write(0, col_idx, tok)

        for row_idx in range(self.shape[0]):
            max_value = torch.max(self[row_idx])
            for col_idx in range(self.shape[1]):
                value = self[row_idx, col_idx].item()
                if value == max_value:
                    worksheet.write(row_idx + 1, col_idx + 1, str(value)[:2+precision], highlight_format)
                elif value == 0.0:
                    worksheet.write(row_idx + 1, col_idx + 1, ".")
                else:
                    worksheet.write(row_idx + 1, col_idx + 1, str(value)[:2+precision])
        workbook.close()


    def test_(self, value):
        return TorchMatrice([[1,2,3], [3,2,1]])

if __name__ == "__main__":
    import doctest
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    doctest.testmod()
    print(f"[DEBUG] doctest finished.\n")
    # from Classes.Snt import Snt
    # t1 = TorchMatrice([[0.1, 0.2, 0.7], [0.4,0.4,0.2], [0.13,0.27,0.6], [0.33,0.33,0.34], [0.30,0.35,0.35]])
    # t1.ecriture_xslt(crt= Snt(3, ['a', 'b', 'c', 'd', 'e']),
    #                     ctx = Snt(2, ['f', 'g', 'h', 'i']),
    #                     absolute_folder= "/home/getalp/lopezfab/Documents/",
    #                     filename="test")

    # t1 = TorchMatrice([[0.1, 0.2, 0.7], [0.4,0.4,0.2], [0.13,0.27,0.6], [0.33,0.33,0.34], [0.30,0.35,0.35]])
    # t1.suppr_inf(medium = 'to_uniform')
    # print(t1)

    # print(TorchMatrice([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]).norm_tenseur(medium = "max"))
    # print(TorchMatrice([[1,2,3], [4,5,6], [7,8,9], [10,11,12], [13,14,15]]).fusion_bpe(groupes= [[4,3], [2,1]]))



    # test.norm_tenseur()
    # print(test)
    # print(test.fusion_ligne(0,1, medium='mean'))
