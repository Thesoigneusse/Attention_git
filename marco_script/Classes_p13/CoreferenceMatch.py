from dataclasses import dataclass, asdict
from typing import List
import json


import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git/marco_script')

@dataclass
class CoreferenceMatch():
    """
    
    Object list of triples (as tuples), where elements are respectively:
     - 1. the index of the aligned token in the system input/output sequence
     - 2. the set_id of the mention (the cluster)
     - 3. True if the aligned system token is identical to the token in the gold sentence.
    """
    aligned_token_index: int
    mention_set_id: int
    is_aligned_token_identical_in_gold_sentence: bool


    def toJson(self) -> dict:
        return {'__EditDistance__': asdict(self)}

    @classmethod
    def fromJson(cls, data):
        return cls(**data)



if __name__ == "__main__":
    import doctest; doctest.testmod()
    print(f"[DEBUG] test cleared\n")
    















