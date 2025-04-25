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

    def __iadd__(self, other: 'BaseWord') -> 'BaseWord':
        """Surcharge l'opérateur += pour concaténer les tokens comme des chaînes.

        Args:
            other (BaseWord | str): Mot ou chaîne à concaténer au mot courant.

        Returns:
            BaseWord: L'instance actuelle avec les tokens mis à jour.
        """
        if isinstance(other, BaseWord):
            self.token += other.token
        elif isinstance(other, str):
            self.token += other
        else:
            raise TypeError(f"Cannot add {type(other)} to BaseWord")
        return self

    def __getitem__(self, key):
        """Permet d'utiliser le slicing sur l'attribut token pour retourner un nouvel objet BaseWord.

        Args:
            key (slice | int): Indice ou slice à appliquer sur le token.

        Returns:
            BaseWord: Nouvelle instance de BaseWord avec le token modifié.
        """
        if isinstance(key, slice):
            # Retourne un nouvel objet BaseWord avec le token découpé
            return BaseWord(identifiant=self.identifiant, token=self.token[key])
        elif isinstance(key, int):
            # Retourne un caractère spécifique du token
            return self.token[key]
        else:
            raise TypeError(f"Invalid argument type: {type(key)}")

    

    def copy(self):
        """ Copie le BaseWord courant
        """
        raise NotImplementedError

    def endswith(self, suffix: str) -> bool:
        """Teste si le mot courant se termine par un suffixe donné.

        Args:
            suffix (str): Suffixe à tester

        Returns:
            bool: True si le mot courant se termine par le suffixe, False sinon

        Tests:
        >>> BaseWord(identifiant=1, token="Hello").endswith("lo")
        True
        >>> BaseWord(identifiant=1, token="Hello").endswith("o")
        True
        """
        assert isinstance(suffix, str), f"suffix must be a string, got {type(suffix)}"
        return self.token.endswith(suffix)

    def startswith(self, prefix: str) -> bool:
        """Teste si le mot courant commence par un préfixe donné.

        Args:
            prefix (str): Préfixe à tester

        Returns:
            bool: True si le mot courant commence par le préfixe, False sinon

        Tests:
        >>> BaseWord(identifiant=1, token="Hello").startswith("He")
        True
        >>> BaseWord(identifiant=1, token="Hello").startswith("o")
        False
        """
        assert isinstance(prefix, str), f"prefix must be a string, got {type(prefix)}"
        return self.token.startswith(prefix)

    def is_padding_mark(self, padding_mark: str) -> bool:
        """Teste si le mot courant est un padding mark.

        Args:
            padding_mark (str): padding mark à tester

        Returns:
            bool: True si le mot courant est un padding mark, False sinon

        Tests:
        >>> BaseWord(identifiant=1, token="<pad>").is_padding_mark("<pad>")
        True
        >>> BaseWord(identifiant=1, token="<eos>").is_padding_mark("<pad>")
        False
        >>> BaseWord(identifiant=1, token="Test").is_padding_mark("<pad>")
        False
        """
        assert isinstance(padding_mark, str), f"padding_mark must be a string, got {type(padding_mark)}"
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
    word1 = BaseWord(identifiant=1, token="Hello")
    word2 = BaseWord(identifiant=2, token="World")
    ic(word1 + word2)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print(f"[DEBUG]Doctest passed for BaseWord")


    main()