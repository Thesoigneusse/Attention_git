from dataclasses import dataclass, field
from typing import List

from ClassesTests.BaseMatrice import SelfAttentionMatrice
from ClassesTests.BaseSentence import BaseSentence

@dataclass
class ConcatenationMatrice(SelfAttentionMatrice):
    """A specific implementation of a SelfAttentionMatrice for concatenation mechanisms.
    It inherits from the SelfAttentionMatrice class and adds a list of sentence heads,
    the index of the where the sentences started.
    This is useful for concatenation mechanisms where the sentences are concatenated together
    And where both current and context sentences are the same.
    """

    sentences_heads:List[int] = field(default_factory=list)

    def get_list_sentences(self) -> List[BaseSentence]:
        """Retourne la liste des phrases de la matrice courante.

        Returns:
            List[BaseSentence]: Liste de phrases de la matrice courante.
        """
        sentences = []
        for start in range(len(self.sentences_heads) - 1):
            sentences.append(self.current_sentence[self.sentences_heads[start]: self.sentences_heads[start + 1]])
        return sentences
    
    def get_number_sentences(self) -> int:
        """Retourne le nombre de phrases de la matrice courante.

        Returns:
            int: Nombre de phrases de la matrice courante.
        """
        return len(self.sentences_heads)
    
    def get_context_sentence(self) -> BaseSentence:
        """Retourne la phrase de contexte de la matrice courante.

        Returns:
            BaseSentence: Phrase de contexte de la matrice courante.
        """
        return self.get_list_sentences[:-1]
    

def main():
    from icecream import ic
    ic("main function of ConcatenationMatrice")


    

if __name__ == '__main__':
    import doctest; doctest.testmod()
    print(f"[DEBUG]Doctest passed for BaseMatrice")
    main()


