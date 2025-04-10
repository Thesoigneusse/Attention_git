from dataclasses import dataclass, field
from typing import List

@dataclass
class Sentence():
    identifiant: int
    tokens: List[str] = field(default_factory=list)

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

    def suppression_padding(self, list_index: List[int] = None, padding_mark='<pad>', strict=True) -> List[int]:
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

    def merge_bpes(self, list_groupes_index: List[List[int]] = None, separator: str = "@@") -> List[List[int]]:
        """Merge les tokens BPEs et retourne une liste contenant les index supprimés

        Args:
            list_index (List[List[int]]): Liste décroissante de groupe d'index à supprimer.
            separator (str, optional): Séparateur des tokens BPEs. Defaults to "@@".

        Returns:
            List[List[int]]: Liste de groupe d'index dont la position est à supprimer

        Examples:
        >>> s1 = Sentence(identifiant= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> list_index = Sentence.list_suppr_pad(s1.tokens, padding_mark="<pad>", strict=False)
        >>> s1.merge_bpes()
        [[9, 8], [4, 3]]
        """

        # Si on n'a pas les index, on les calcules
        if list_groupes_index is None:
            list_groupes_index = Sentence.list_merge_bpe(self.tokens, separator=separator)
        
        # Pour chaque groupe d'index à fusionner
        for groupe in list_groupes_index:
            # On parcours les tokens du groupe à partir du 2ème token
            for i in groupe[1:]:
                # On fusionne le token courant au premier token du groupe et on supprime le token courant
                self.tokens[groupe[0]] = self.tokens[i][:-len(separator)] + self.tokens[groupe[0]]
                del self.tokens[i]

        return list_groupes_index

    @staticmethod
    def list_suppr_pad(tokens: List[str], padding_mark="<pad>", strict=False)-> List[int]:
        """retourne la liste des index du padding à supprimer par ordre décroissant.

        Args:
            tokens (List[str]): Liste de tokens
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

    @staticmethod
    def list_merge_bpe(tokens: List[str], separator: str = "@@")-> List[int]:
        """retourne la liste des index des tokens BPEs à supprimer par ordre décroissant.

        Args:
            separator (str, optional): Séparateur des tokens BPEs. Defaults to "@@".

        Returns:
            List[int]: Liste d'index dont la position est à supprimer

        Example
        >>> Sentence.list_merge_bpe(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        [[9, 8], [4, 3]]
        >>> Sentence.list_merge_bpe(["Ce@@", "ci", "te@@", "st", "."])
        [[3, 2], [1, 0]]
        """
        # Liste des groupes de BPEs à merge
        list_merge_bpe = []
        # Indique si le token précédant était aussi un BPE
        is_previous_tok_bpe = False

        # On parcours les index des tokens dans le sens décroissant
        # Pour chaque token
        for i in range(len(tokens)-2, -1, -1):
            # Si le token se termine par le séparateur et que le token précédant n'est pas un BPE
            if tokens[i-1].endswith(separator) and not is_previous_tok_bpe:
                # On ajoute un groupe d'index contenant l'index du token courant à la liste des tokens à merge
                list_merge_bpe.append([i]) 
                is_previous_tok_bpe = True
            # Si le token se termine par le séparateur et que le token précédant est un BPE
            elif tokens[i-1].endswith(separator) and is_previous_tok_bpe:
                # On ajoute l'index du token courant au groupe précédant
                list_merge_bpe[-1].append(i)
            # Si le token ne se termine pas par le séparateur et que le token précédant est un BPE
            elif not tokens[i-1].endswith(separator) and is_previous_tok_bpe:
                # On ajoute l'index au groupe précédant
                # et on indique que le token précédant n'est pas un BPE
                list_merge_bpe[-1].append(i)
                is_previous_tok_bpe = False
            elif not tokens[i-1].endswith(separator) and not is_previous_tok_bpe:
                is_previous_tok_bpe = False
        return list_merge_bpe

def main():
    from icecream import ic
    print("main function")
    s1 = Sentence(identifiant= 3,
                  tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
    s2 = Sentence(identifiant= 4,
                  tokens=["Ce@@", "ci", "te@@", "st", "."])
    ic(s1)
    ic(Sentence.list_merge_bpe(s1.tokens))
    ic(s1)
    ic(Sentence.list_merge_bpe(s2.tokens))
    ic(s2.merge_bpes())
    ic(s2)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print(f"[DEBUG]Doctest passed for Sentence")


    main()