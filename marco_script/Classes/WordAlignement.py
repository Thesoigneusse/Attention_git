from dataclasses import dataclass, asdict
from typing import Literal
import json

# import sys
# sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/marco_script')

@dataclass
class WordAlignement():
    """Objet representant un alignement entre un mot de reference et un mot hypothese. 
    Specifie:
    - le alignement_type d'alignement (ins, del, sub, match), 
    - l'index du mot dans la phrase de reference
    - l'index du mot dans la phrase hypothese.
    """
    index_reference: int
    index_hypothese: int
    alignement_type: Literal['del', 'ins', 'sub', 'match']
    
    def __init__(self, alignement_type: str, index_reference: int, index_hypothese: int):
        self.alignement_type = alignement_type
        self.index_reference = index_reference
        self.index_hypothese = index_hypothese

    @property
    def alignement_type(self):
        return self._alignement_type
    @alignement_type.setter
    def alignement_type(self, value):
        assert value in ['ins', 'del', 'sub', 'match'], f"[DEBUG] value must be in ['ins', 'del', 'sub', 'match']. Current alignement_type: {value}"
        self._alignement_type = value

    def toJson(self) -> dict:
        return asdict(self)

    def to_list(self) -> list:
        """Retourne l'alignement du mot entre la phrase de référence et la phrase hypothèse sous forme de liste (ancien format de marco)

        Returns:
            list: Liste contenant les informations sur l'alignement entre le mot de la phrase de référence et le mot de la phrase hypothèse.
                0: type d'alignement
                1: index du mot dans la phrase de référence
                2: index du mot dans la phrase hypothèse
        
        Tests:
        >>> WordAlignement("ins", 1, 2).to_list()
        ['ins', 1, 2]
        """
        return [self.alignement_type, self.index_reference, self.index_hypothese]

if __name__ == "__main__":
    import doctest; doctest.testmod()
    print(f"[debug] doctest cleared\n")
    alignement = WordAlignement("ins", 1, 2)
    print(alignement.toJson())

