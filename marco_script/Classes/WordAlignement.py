from dataclasses import dataclass, field
from typing import Literal
import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/marco_script')

@dataclass
class WordAlignement():
    """Objet representant un alignement entre un mot de reference et un mot hypothese. 
    Specifie:
    - le alignement_type d'alignement (ins, del, sub, match), 
    - l'index du mot dans la phrase de reference
    - l'index du mot dans la phrase hypothese.
    """
    alignement_type: Literal['del', 'ins', 'sub', 'match']
    index_reference: int
    index_hypothese: int
    
    def __init__(self, alignement_type: str, index_reference: int, index_hypothese: int):
        self.alignement_type = alignement_type
        self.index_reference = index_reference
        self.index_hypothese = index_hypothese

    def __post_init__(self):
        """On restreint les valeurs possibles de alignement_type à 'del', 'ins', 'sub', 'match'

        Raises:
            ValueError: _description_
        """
        alignement_type_valeurs_valides = {'del', 'ins', 'sub', 'match'}
        if self.alignement_type not in alignement_type_valeurs_valides:
            raise ValueError(f"alignement_type invalide : {self.alignement_type}. Possible values: {alignement_type_valeurs_valides}")

    # def __str__(self):
    #     return f"Alignement(alignement_type={self.alignement_type}, index_reference={self.index_reference}, index_hypothese={self.index_hypothese})"
    
    # def __repr__(self):
    #     return str(self)




if __name__ == "__main__":
    alignement = WordAlignement("ins", 1, 2)
    print(alignement)

