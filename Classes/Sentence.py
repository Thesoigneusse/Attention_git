from dataclasses import dataclass, field
from typing import List

@dataclass
class Sentence():
    identifiant: int
    tokens: List[str] = field(default_factory=list)
    raw_length: int

    def __len__(self):
        """Retourne la longueur de la phrase courante en nombre de tokens.

        Returns:
            int: nombre de tokens dans la phrase courante
        """
        return len(self.tokens)
    
    def __add__(self, other: 'Sentence') -> 'Sentence':
        """Additionne 2 phrases en concaténant les tokens.
            Supported type: 
             - Sentence : conserve le self.identifiant et concatène les tokens self.tokens + other.tokens
             - List[str] : conserve le self.identifiant et concatène les tokens self.tokens + other

        Args:
            other (Sentence): Sentence à concaténer à la suite de la Sentence courante

        Returns:
            Sentence: New instance of Sentence with the concatenated tokens
        """
        assert isinstance(other, Sentence) \
            or (isinstance(other, list) and all([isinstance(val, str) for val in other])), \
            f"other must be an instance of Snt or a List[str]. Current type: {type(other)}"
        if isinstance(other, Sentence):
            return Sentence(identifiant=self.identifiant, tokens=self.tokens + other.tokens)
        elif isinstance(other, list):
            return Snt(identifiant=self.identifiant, tokens=self.tokens + other)
