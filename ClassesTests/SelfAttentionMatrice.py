from dataclasses import dataclass, field
from typing import List

from ClassesTests.BaseMatrice import BaseMatrice


@dataclass
class SelfAttentionMatrice(BaseMatrice):
    """A specific implementation of a Attention Matrice for self-attention mechanisms.
    It inherits from the BaseMatrice class and merge the context sentence attribute with the current sentence attribute.
    This is useful for self-attention mechanisms where the context and current sentences are the same.
    """

    def __post__init__(self):
        self.context_sentence = self.current_sentence