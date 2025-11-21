from dataclasses import dataclass, field, asdict
from typing import List

@dataclass
class SystemOutput():

    current_sentence: str # 0
    context_sentence: str # 1
    attention_matrice: List[List[float]] = field(default_factory=list) # 2

    def toJson(self) -> dict:
        return {'__SystemOutput__': asdict(self)}

    @classmethod
    def fromJson(cls, data):
        return cls(**data)
