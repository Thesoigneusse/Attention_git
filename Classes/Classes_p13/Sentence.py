from typing import List, Optional, Union
import copy
import json


class Sentence:
    """
    Représente une phrase composée de tokens et identifiée par un identifiant unique.

    Attributes:
        identifiant (int): identifiant unique de la phrase (>=0 ou -1 pour dummy)
        tokens (List[str]): liste des tokens composant la phrase

    Examples:
        >>> s = Sentence(identifiant=3, tokens=["Ce@@", "ci", "est", "un", "te@@", "st", ".", "<eos>"])
        >>> s.identifiant
        3
        >>> s.tokens
        ['Ce@@', 'ci', 'est', 'un', 'te@@', 'st', '.', '<eos>']
    """

    def __init__(self, identifiant: Optional[int] = None, tokens: Optional[List[str]] = None):
        """
        Initialise une Sentence.

        Args:
            identifiant (Optional[int]): identifiant unique (>=0 ou -1)
            tokens (Optional[List[str]]): liste de tokens
        """
        self.identifiant = identifiant
        self.tokens = tokens or []

    def __repr__(self) -> str:
        """Représentation officielle de l'objet."""
        return f"Sentence(id={self.identifiant}, tokens={self.tokens})"

    def __str__(self) -> str:
        """Représentation lisible de l'objet en JSON."""
        # return json.dumps(self.to_dict(), indent=4, sort_keys=True)
        return self.__repr__()

    def __len__(self) -> int:
        """Renvoie le nombre de tokens dans la Sentence."""
        return len(self.tokens)

    def __add__(self, other: Union['Sentence', List[str]]) -> 'Sentence':
        """
        Concatène une autre Sentence ou une liste de tokens à cette Sentence. Garde l'identifiant de la première Sentence

        Args:
            other (Sentence | List[str]): Sentence ou liste de tokens à concaténer

        Returns:
            Sentence: nouvelle Sentence résultante
        """
        if isinstance(other, Sentence):
            return Sentence(identifiant=self.identifiant, tokens=self.tokens + other.tokens)
        elif isinstance(other, list) and all(isinstance(tok, str) for tok in other):
            return Sentence(identifiant=self.identifiant, tokens=self.tokens + other)
        else:
            raise TypeError("Other must be a Sentence or a List[str]")

    def __mul__(self, other: int) -> List['Sentence']:
        """
        Duplique la Sentence plusieurs fois.

        Args:
            other (int): nombre de duplications (doit être >0)

        Returns:
            List[Sentence]: liste de copies de la Sentence
        """
        if not isinstance(other, int) or other <= 0:
            raise ValueError("Multiplication requires a positive integer")
        return [copy.deepcopy(self) for _ in range(other)]

    def copy(self) -> 'Sentence':
        """
        Retourne une copie profonde de la Sentence.

        Returns:
            Sentence: copie de la Sentence
        """
        return copy.deepcopy(self)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def tokens(self) -> List[str]:
        """Getter des tokens."""
        return self._tokens

    @tokens.setter
    def tokens(self, tokens: List[str]) -> None:
        """Setter des tokens.

        Args:
            tokens (List[str]): liste de tokens
        """
        if tokens is not None:
            if not isinstance(tokens, list):
                raise TypeError(f"tokens must be a list. Got {type(tokens)}")
            if not all(isinstance(tok, str) for tok in tokens):
                raise TypeError(f"All tokens must be strings. Got {[type(tok) for tok in tokens]}")
            self._tokens = copy.copy(tokens)
        else:
            self._tokens = []

    @property
    def identifiant(self) -> Optional[int]:
        """Getter de l'identifiant."""
        return self._identifiant

    @identifiant.setter
    def identifiant(self, identifiant: Optional[int]) -> None:
        """Setter de l'identifiant.

        Args:
            identifiant (int | None): identifiant unique (>=0 ou -1)
        """
        if identifiant is not None and not isinstance(identifiant, int):
            raise TypeError(f"identifiant must be an int. Got {type(identifiant)}")
        self._identifiant = identifiant

    # ------------------------------------------------------------------
    # Static Utilities
    # ------------------------------------------------------------------
    @staticmethod
    def list_suppr_pad(tokens: List[str], padding_mark: str = "<pad>", strict: bool = False) -> List[int]:
        """
        Retourne les indices des tokens de padding à supprimer.

        Args:
            tokens (List[str]): liste de tokens
            padding_mark (str): token représentant le padding
            strict (bool): si True, supprime tout padding sinon garde un padding

        Returns:
            List[int]: indices des tokens à supprimer
        """
        stop = -1 if strict else 0
        return [i for i in range(len(tokens) - 1, stop, -1) if tokens[i] == padding_mark]

    @staticmethod
    def list_fusion_bpe(tokens: List[str], BPE_mark: str = "@@") -> Optional[List[List[int]]]:
        """
        Retourne les indices des tokens BPE à fusionner (ordre décroissant).

        Args:
            tokens (List[str]): liste de tokens
            BPE_mark (str): suffixe indiquant un token BPE

        Returns:
            Optional[List[List[int]]]: liste de groupes d'indices à fusionner
        """
        if not tokens or tokens[-1].endswith(BPE_mark):
            raise ValueError("Dernier token ne doit pas contenir de BPE_mark")

        indices = [i for i, tok in enumerate(tokens) if tok.endswith(BPE_mark)]
        if not indices:
            return None

        groupes = []
        current_group = [indices[0]]

        for i in indices[1:]:
            if i == current_group[-1] + 1:
                current_group.append(i)
            else:
                groupes.append(current_group)
                current_group = [i]
        groupes.append(current_group)

        # Ajouter le token final après le dernier BPE
        for grp in groupes:
            end = grp[-1] + 1
            if end < len(tokens) and not tokens[end].endswith(BPE_mark):
                grp.append(end)

        # Renverser les groupes et les indices pour avoir ordre décroissant
        return [list(reversed(grp)) for grp in reversed(groupes)]

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def append(self, value: Union[str, List[str]]) -> None:
        """
        Ajoute un ou plusieurs tokens à la fin de la phrase.

        Args:
            value (str | List[str]): token(s) à ajouter
        
        Example:
            >>> t = Sentence(identifiant=2, tokens= ["a", "b", "c", "d", "e"])
            >>> t.append('f')
            >>> t
            Sentence(id=2, tokens=['a', 'b', 'c', 'd', 'e', 'f'])
            >>> t = Sentence(identifiant=2, tokens= ["a", "b", "c", "d", "e"])
            >>> t.append(["f", "g"])
            >>> t
            Sentence(id=2, tokens=['a', 'b', 'c', 'd', 'e', 'f', 'g'])


        """
        if isinstance(value, str):
            self.tokens.append(value)
        elif isinstance(value, list) and all(isinstance(tok, str) for tok in value):
            self.tokens.extend(value)
        else:
            raise TypeError("Value must be a string or list of strings")

    def insert(self, index: int, value: str) -> None:
        """
        Insère un token à une position donnée.

        Args:
            index (int): position d'insertion
            value (str): token à insérer
        """
        if not isinstance(index, int):
            raise TypeError("index must be an int")
        if not isinstance(value, str):
            raise TypeError("value must be a str")
        self.tokens.insert(index, value)

    def suppr_pad(self, list_index: Optional[List[int]] = None, padding_mark: str = "<pad>", strict: bool = False) -> List[int]:
        """
        Supprime les tokens de padding dans la phrase.

        Args:
            list_index (Optional[List[int]]): indices à supprimer (None pour calculer automatiquement)
            padding_mark (str): token représentant le padding
            strict (bool): si True supprime tout padding sinon garde un padding

        Returns:
            List[int]: indices supprimés
        """
        list_index = list_index or self.list_suppr_pad(self.tokens, padding_mark=padding_mark, strict=strict)
        for i in list_index:
            del self.tokens[i]
        return list_index

    def fusion_bpe(self, list_bpe: Optional[List[List[int]]] = None, BPE_mark: str = "@@") -> Optional[List[List[int]]]:
        """
        Fusionne les tokens BPE de la phrase.

        Args:
            list_bpe (Optional[List[List[int]]]): groupes d'indices BPE à fusionner (None pour calculer automatiquement)
            BPE_mark (str): suffixe indiquant un token BPE

        Returns:
            Optional[List[List[int]]]: groupes d'indices fusionnés
        """
        groupes_bpe = [sorted(liste) for liste in list_bpe, reverse=True)] or self.list_fusion_bpe(self.tokens, BPE_mark=BPE_mark)
        if groupes_bpe is not None:
            for grp in groupes_bpe:
                if len(grp) < 2:
                    continue
                for i in range(grp[1], grp[-1] - 1, -1):
                    self.tokens[i] = f"{self.tokens[i].split(BPE_mark)[0]}{self.tokens[i+1]}"
                    del self.tokens[i+1]
        return groupes_bpe

    def to_dict(self) -> dict:
        """Retourne un dictionnaire représentant la phrase."""
        return {'identifiant': self.identifiant, 'tokens': self.tokens}

    def toJSON(self) -> str:
        """Retourne un JSON représentant la phrase."""
        return json.dumps(self.to_dict(), indent=4, sort_keys=True)


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    test = Sentence(identifiant=2, tokens= ["a", "b", "c", "d", "e"])
    test.append(["f", "g"])
    print(test)

