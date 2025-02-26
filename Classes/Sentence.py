from dataclasses import dataclass, field
from typing import List

@dataclass
class Sentence():
    identifiant: int
    tokens: List[str] = field(default_factory=list)
    raw_length: int
    fusion_BPE_strategy: 

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

    def copy(self):
        from copy import copy
        return copy(self)

    def suppression_padding(self, list_index: List[int] = None, padding_mark='<pad>', strict=False) -> List[int]:
        """Supprime le padding de la phrase et retourne une liste contenant les index supprimés

        Args:
            list_index (List[int], optional): Liste décroissante d'index à supprimer. Defaults to None.
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to False.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        
        Examples:
        >>> s1 = Sentence(identifiant= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> list_index = Sentence.list_suppr_pad(s1.tokens, padding_mark="<pad>", strict=False)
        >>> s1.suppression_padding(list_index)
        [6, 2, 1]
        >>> s2 = Sentence(identifiant= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> s2.suppression_padding(strict=True)
        [6, 2, 1, 0]
        """
        if list_index is None:
            # Si la liste des index  correspondant aux tokens de padding n'est pas fournie, on la génère
            list_index = Sentence.list_suppr_pad(self.tokens, padding_mark=padding_mark, strict=strict)

        # On supprime les tokens de padding
        for i in list_index:
            del self.tokens[i]
        return list_index

    @staticmethod
    def list_suppr_pad(tokens, padding_mark="<pad>", strict=False)-> List[int]:
        """retourne la liste des index du padding à supprimer par ordre décroissant.

        Args:
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to False.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        Example
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="<pad>", strict=False)
        [6, 2, 1]
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="<pad>", strict=True)
        [6, 2, 1, 0]
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="ci", strict=True)
        [4]
        >>> Sentence.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="padding", strict=True)
        []
        """
        stop = -1 if strict else 0
        list_suppr_pad = []
        for i in range(len(tokens)-1, stop, -1):
            if tokens[i] == padding_mark:
                list_suppr_pad.append(i)
        return list_suppr_pad

def main():
    print("main function")


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    main()