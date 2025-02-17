from dataclasses import dataclass, field, asdict
from typing import List
import json

import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/marco_script')
from Classes.WordAlignement import WordAlignement

from typing import TYPE_CHECKING
from icecream import ic


@dataclass
class SentenceAlignement():
    """
0. The NMT system input/output sentence : type = str (exemple: '"Even in purely non-religious terms, homosexuality represents a misuse of the sexual faculty.')
1. The corresponding raw ParCorFull 2 corpus sentence : type = str (exemple: '" Even in purely non-religious terms , homosexuality represents a misuse of the sexual faculty .')
2. The sequence of uniq token identifiers for tokens in the corpus sentence : type = List[str] (exemple: ['000_1756-word_1', '000_1756-word_2', '000_1756-word_3', '000_1756-word_4', '000_1756-word_5', '000_1756-word_6', '000_1756-word_7', '000_1756-word_8', '000_1756-word_9', '000_1756-word_10', '000_1756-word_11', '000_1756-word_12', '000_1756-word_13', '000_1756-word_14', '000_1756-word_15', '000_1756-word_16'])
3. The corresponding tokenized ParCorFull 2 corpus sentence : type = List[str] (exemple: [('del', 0, None), ('sub', 1, 0), ('match', 2, 1), ('match', 3, 2), ('match', 4, 3), ('del', 5, None), ('sub', 6, 4), ('match', 7, 5), ('match', 8, 6), ('match', 9, 7), ('match', 10, 8), ('match', 11, 9), ('match', 12, 10), ('match', 13, 11), ('del', 14, None), ('sub', 15, 12)])
4. The same sentence as in 1. annotated with mentions, in the form of tokens "decorated" with [<token>]set_idxxx : type = str (exemple: '#[``]#-set_219 #[Even]#-set_219 #[in]#-set_219 #[purely]#-set_219 #[non-religious]#-set_219 #[terms]#-set_219 #[,]#-set_219 #[homosexuality]#-set_219 #[represents]#-set_219 #[a]#-set_219 #[misuse]#-set_219 #[of]#-set_219 #[the]#-set_219 #[sexual]#-set_219 #[faculty]#-set_219 #[.]#-set_219')
    """
    identifiant: int
    system_input_sentence: str # 0
    raw_parcorfull_sentence: str # 1
    annotated_system_input_sentence: str # 4
    unique_token_identifiers_sequence: List[str] = field(default_factory=list) # 2
    tokenized_parcorfull_sentence: List[WordAlignement] = field(default_factory=list) # 3



    def __add__(self, other: 'SentenceAlignement') -> 'SentenceAlignement':
        """Concatène 2 SentenceAlignement en concaténant chaque str/list de other à self. identifiant est le self.identifiant s'il est différent de None.

        Args:
            other (SentenceAlignement): une autre SentenceAlignement à concaténer après le self

        Raises:
            NotImplementedError: si l'addition avec le type SentenceAlignement n'est pas implémentée

        Returns:
            SentenceAlignement: nouvelle instance de SentenceAlignement
        
        Tests:
            >>> sa1 = SentenceAlignement(identifiant=1, system_input_sentence="Hello", raw_parcorfull_sentence="Bonjour", annotated_system_input_sentence="Hi", tokenized_parcorfull_sentence=["Bonjour"], unique_token_identifiers_sequence=[1])
            >>> sa2 = SentenceAlignement(identifiant=2, system_input_sentence="World", raw_parcorfull_sentence="Monde", annotated_system_input_sentence="Hey", tokenized_parcorfull_sentence=["Monde"], unique_token_identifiers_sequence=[2])
            >>> sa3 = sa1 + sa2
            >>> sa3.identifiant
            1
            >>> sa3.system_input_sentence
            'HelloWorld'
            >>> sa3.raw_parcorfull_sentence
            'BonjourMonde'
            >>> sa3.annotated_system_input_sentence
            'HiHey'
            >>> sa3.tokenized_parcorfull_sentence
            ['Bonjour', 'Monde']
            >>> sa3.unique_token_identifiers_sequence
            [1, 2]
        """
        if isinstance(other, SentenceAlignement):
            ic(self)
            ic(other)
            return SentenceAlignement(identifiant= self.identifiant if self.identifiant is not None else other.identifiant,
                                      system_input_sentence= self.system_input_sentence + other.system_input_sentence,
                                      raw_parcorfull_sentence= self.raw_parcorfull_sentence + other.raw_parcorfull_sentence,
                                      annotated_system_input_sentence= self.annotated_system_input_sentence + other.annotated_system_input_sentence,
                                      tokenized_parcorfull_sentence= self.tokenized_parcorfull_sentence + other.tokenized_parcorfull_sentence,
                                      unique_token_identifiers_sequence= self.unique_token_identifiers_sequence + other.unique_token_identifiers_sequence)
        raise NotImplementedError(f"[debug] Addition between SentenceAlignement and {type(other)} not implemented")

    def toJson(self) -> dict:
        return {'__SentenceAlignement__': asdict(self)}

    @classmethod
    def fromJson(cls, data):
        return cls(**data)



if __name__ == "__main__":
    import doctest; doctest.testmod()
    print(f"[debug] doctest cleared\n")

    sentence = SentenceAlignement(
        identifiant=1,
        system_input_sentence='"Even in purely non-religious terms, homosexuality represents a misuse of the sexual faculty."',
        raw_parcorfull_sentence='" Even in purely non-religious terms , homosexuality represents a misuse of the sexual faculty ."',
        unique_token_identifiers_sequence=['000_1756-word_1'],
        tokenized_parcorfull_sentence=[WordAlignement('del', 0, None)],
        annotated_system_input_sentence='#["Even"]#-set_219 #["in"]#-set_219 #["purely"]#-set_219 #["non-religious"]#-set_219 #["terms"]#-set_219 #[,]#-set_219 #["homosexuality"]#-set_219 #["represents"]#-set_219 #["a"]#-set_219 #["misuse"]#-set_219 #["of"]#-set_219 #["the"]#-set_219 #["sexual"]#-set_219 #["faculty"]#-set_219 #["."]#-set_219'
    )
    sentence2 = SentenceAlignement(
        identifiant=2,
        system_input_sentence='"Even in purely non-religious terms, homosexuality represents a misuse of the sexual faculty."',
        raw_parcorfull_sentence='" Even in purely non-religious terms , homosexuality represents a misuse of the sexual faculty ."',
        unique_token_identifiers_sequence=['000_1756-word_3'],
        tokenized_parcorfull_sentence=[WordAlignement('ins', 0, None)],
            
        annotated_system_input_sentence='#["Even"]#-set_219 #["in"]#-set_219 #["purely"]#-set_219 #["non-religious"]#-set_219 #["terms"]#-set_219 #[,]#-set_219 #["homosexuality"]#-set_219 #["represents"]#-set_219 #["a"]#-set_219 #["misuse"]#-set_219 #["of"]#-set_219 #["the"]#-set_219 #["sexual"]#-set_219 #["faculty"]#-set_219 #["."]#-set_219'
    )
    print(sentence.toJson())
    print('**************************')
    print(sentence2.toJson())