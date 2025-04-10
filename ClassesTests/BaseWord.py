from dataclasses import dataclass
from typing import List

@dataclass
class BaseWord():
    identifiant: int
    token: str

    def __len__(self):
        """Retourne la longueur du mot courante en nombre de caractère.

        Returns:
            int: nombre de caractères dans le mot courant
        """
        return len(self.token)
    
    def __add__(self, other: 'BaseWord') -> 'BaseWord':
        """Additionne 2 tokens en concaténant les caractères.

        Args:
            other (BaseWord): Mot à concaténer à la suite du mot courant

        Returns:
            BaseWord: New instance of BaseWord with the concatenated tokens
        """
        if isinstance(other, BaseWord):
            return BaseWord(identifiant=self.identifiant, token=self.token + other.token)
        elif isinstance(other, str):
            return BaseWord(identifiant=self.identifiant, token=self.token + other)
        raise TypeError(f"Cannot add {type(other)} to BaseWord")

    def copy(self):
        """ Copie le BaseWord courant
        """
        raise NotImplementedError

    def is_padding_mark(self, padding_mark: str) -> bool:
        """Teste si le mot courant est un padding mark.

        Args:
            padding_mark (str): padding mark à tester

        Returns:
            bool: True si le mot courant est un padding mark, False sinon

        Tests:
        >>> BaseWord(identifiant=1, token="<pad>").is_padding_mark(BaseWord(identifiant=-1, token="<pad>"))
        True
        >>> BaseWord(identifiant=1, token="<eos>").is_padding_mark(BaseWord(identifiant=-1, token="<pad>"))
        False
        >>> BaseWord(identifiant=1, token="Test").is_padding_mark(BaseWord(identifiant=-1, token="<pad>"))
        False
        """
        return self.token == padding_mark

    def is_bpe(self, bpe_mark:str) -> bool:
        """Teste si le mot courant est un BPE.

        Args:
            bpe_mark (str): BPE à tester

        Returns:
            bool: True si le mot courant est un BPE, False sinon
            
        Tests:
        >>> BaseWord(identifiant=1, token="Ce@@").is_bpe("@@")
        True
        >>> BaseWord(identifiant=1, token="Ce").is_bpe("@@")
        False
        >>> BaseWord(identifiant=1, token="Ce@@").is_bpe("Ce")
        False
        """
        return self.token.endswith(bpe_mark)
   
    def load_word(self, identifiant: int, token: str) -> 'BaseWord':
        """Charge un mot de la classe BaseWord.

        Args:
            identifiant (int): Identifiant du mot
            token (str): Mot à charger

        Returns:
            BaseWord: Instance de la classe BaseWord
        """
        return BaseWord(identifiant=identifiant, token=token)

def main():
    from icecream import ic
    print("main function")
    ic("BaseWord class main function")

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print(f"[DEBUG]Doctest passed for BaseWord")


    main()