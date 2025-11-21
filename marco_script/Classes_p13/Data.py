from typing import TypedDict, Dict, List, Tuple
from Classes_p13.WordAlignement import WordAlignement
from Classes_p13.Coref import Coref

class Data(TypedDict):
    text: Dict[str, Tuple[str, List[WordAlignement]]]
    words: Dict[str, str | None]
    coref: List[Coref]
    words_in_coref: Dict[str, str]


