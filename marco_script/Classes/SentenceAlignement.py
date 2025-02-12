from dataclasses import dataclass, field, asdict
from typing import List
import json

import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/marco_script')
from Classes.WordAlignement import WordAlignement

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
    system_input_sentence: str
    raw_parcorfull_sentence: str
    annotated_system_input_sentence: str
    unique_token_identifiers_sequence: List[str] = field(default_factory=list)
    tokenized_parcorfull_sentence: List[WordAlignement] = field(default_factory=list)

    # def __post_init__(self):
    #     self.unique_token_identifiers_sequence = []

    def toJson(self) -> dict:
        return asdict(self)

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