import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/marco_script')

class WordAlignement():
    """Objet representant un alignement entre un mot de reference et un mot hypothese. 
    Specifie:
    - le alignement_type d'alignement (ins, del, sub, match), 
    - l'index du mot dans la phrase de reference
    - l'index du mot dans la phrase hypothese.
    """
    
    def __init__(self, alignement_type: str, index_reference: int, index_hypothese: int):
        self.alignement_type = alignement_type
        self.index_reference = index_reference
        self.index_hypothese = index_hypothese

    def __str__(self):
        return f"Alignement(alignement_type={self.alignement_type}, index_reference={self.index_reference}, index_hypothese={self.index_hypothese})"

    def __repr__(self):
        return str(self)

    @property
    def alignement_type(self):
        return self._alignement_type
    @alignement_type.setter
    def alignement_type(self, alignement_type):
        assert alignement_type in ['ins', 'del', 'sub', 'match'], f"[DEBUG] alignement_type must be in ['ins', 'del', 'sub', 'match']. Current alignement_type: {alignement_type}"
        self._alignement_type = alignement_type

    @property
    def index_reference(self):
        return self._index_reference
    @index_reference.setter
    def index_reference(self, index_reference):
        self._index_reference = index_reference

    @property
    def index_hypothese(self):
        return self._index_hypothese
    @index_hypothese.setter
    def index_hypothese(self, index_hypothese):
        self._index_hypothese = index_hypothese

if __name__ == "__main__":
    alignement = WordAlignement("ins", 1, 2)
    print(alignement)