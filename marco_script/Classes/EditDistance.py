from dataclasses import dataclass, asdict
from typing import List
import json


import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/marco_script')
from Classes.WordAlignement import WordAlignement

@dataclass
class EditDistance():
    """Objet permettant de calculer la edit distance entre deux phrases.
        Contient les informations suivantes:
            - nombre_erreur_insertion (int): Nombre d'erreurs d'insertion
            - nombre_erreur_suppression (int): Nombre d'erreurs de suppression
            - nombre_erreur_substitution (int): Nombre d'erreurs de substitution
            - taille_tenseur_reference (int): Taille du tenseur de référence
            - alignements (List[WordAlignement]): Liste des alignements entre les deux
    """
    nombre_erreur_insertion: int
    nombre_erreur_suppression: int
    nombre_erreur_substitution: int
    taille_tenseur_reference: int
    alignements: List[WordAlignement]


    def __init__(self, 
                 nombre_erreur_insertion: int, 
                 nombre_erreur_suppression: int, 
                 nombre_erreur_substitution: int, 
                 taille_tenseur_reference: int, 
                 alignements: List[WordAlignement]):
        
        self.nombre_erreur_insertion = nombre_erreur_insertion
        self.nombre_erreur_suppression = nombre_erreur_suppression
        self.nombre_erreur_substitution = nombre_erreur_substitution
        self.taille_tenseur_reference = taille_tenseur_reference
        self.alignements = alignements
    
    @property
    def alignements(self):
        return self._alignements
    @alignements.setter
    def alignements(self, alignements):
        assert isinstance(alignements, list), "[DEBUG] alignements must be a list. Current type: {}".format(type(alignements))
        if len(alignements) > 0:
            assert all(isinstance(alignement, WordAlignement) for alignement in alignements), "[DEBUG] alignements must be a list of WordAlignement. Current list: {}".format([type(alignement) for alignement in alignements])
        self._alignements = alignements

    def get_wer(self) -> float:
        """Retourne le Word Error Rate (WER) de l'alignement.

        Returns:
            float: Valeur du WER pour l'alignement donné
        """
        return float(sum([self.nombre_erreur_insertion, self.nombre_erreur_suppression, self.nombre_erreur_substitution])) / self.taille_tenseur_reference

    def __add__(self, other):
        from copy import deepcopy
        """Permet de sommer deux objets EditDistance en additionnant les valeurs des attributs.
        alignements étant une liste on retourne self.alignements + other.alignements.

        Args:
            other (EditDistance): une autre instance de EditDistance

        Returns:
            EditDistance: addition des deux objets EditDistance

        Tests:
        >>> ed = EditDistance(1, 2, 3, 4, [WordAlignement("ins", 1, 2), WordAlignement("del", 2, 3)])
        >>> ed2 = EditDistance(5, 6, 7, 8, [WordAlignement("ins", 1, 2), WordAlignement("del", 4, 5)])
        >>> print(ed + ed2)
        EditDistance(nombre_erreur_insertion=6, nombre_erreur_suppression=8, nombre_erreur_substitution=10, taille_tenseur_reference=12, alignements=[WordAlignement(alignement_type=ins, index_reference=1, index_hypothese=2), WordAlignement(alignement_type=del, index_reference=2, index_hypothese=3), WordAlignement(alignement_type=ins, index_reference=1, index_hypothese=2), WordAlignement(alignement_type=del, index_reference=4, index_hypothese=5)])
        """
        assert isinstance(other, EditDistance), "[DEBUG] other must be an EditDistance object. Current type: {}".format(type(other))
        return EditDistance(self.nombre_erreur_insertion + other.nombre_erreur_insertion, 
                             self.nombre_erreur_suppression + other.nombre_erreur_suppression, 
                             self.nombre_erreur_substitution + other.nombre_erreur_substitution, 
                             self.taille_tenseur_reference + other.taille_tenseur_reference, 
                             self.alignements + other.alignements)

    def toJson(self) -> dict:
        return {'__EditDistance__': asdict(self)}

    @classmethod
    def fromJson(cls, data):
        return cls(**data)



if __name__ == "__main__":
    import doctest; doctest.testmod()
    print(f"[DEBUG] test cleared\n")
    
    alignements = [WordAlignement("ins", 1, 2), WordAlignement("del", 2, 3)]
    ed = EditDistance(1, 2, 3, 4, alignements)
    ed2 = EditDistance(5, 6, 7, 8, alignements)
    with open("./test.json", "w") as f:
        f.write(ed.toJson())
    print(ed.toJson())
















