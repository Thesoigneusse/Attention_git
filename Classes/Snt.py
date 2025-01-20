from typing import List
import copy


class Snt:
    def __init__(self, identifiant: int = None, tokens: List[str] = None):
        """Représente une phase.

        Args:
            identifiant (int): identifiant de la phrase (must be >= 0 or ==-1)
            tokens (List[str]): liste de str contenus dans la phrase
        
        Example
        >>> s1 = Snt(identifiant= 3, tokens= ["Ce@@", "ci", "est", "un", "te@@", "st", ".", "<eos>"])
        >>> s1.identifiant
        3
        >>> s1.tokens
        ['Ce@@', 'ci', 'est', 'un', 'te@@', 'st', '.', '<eos>']
        """
        self.identifiant = identifiant
        self.tokens = tokens

    def __repr__(self):
        return self.__json__()

    def __str__(self):
        return str(self.__json__())
        # return f"Snt(id={self.identifiant}, tokens={self.tokens})"

    def __len__(self):
        return len(self.tokens)
    
    def __json__(self):
        return str(self.__dict__)

    def __add__(self, other):
        assert isinstance(other, Snt) or (isinstance(other, list) and all([isinstance(val, str) for val in other])), f"other must be an instance of Snt or a List[str]. Current type: {type(other)}"
        if isinstance(other, Snt):
            return Snt(identifiant=self.identifiant, tokens=self.tokens + other.tokens)
        elif isinstance(other, list):
            return Snt(identifiant=self.identifiant, tokens=self.tokens + other)

    def __mul__(self, other):
        assert isinstance(other, int), f"[DEBUG] operator __mul__ only supported on positive integers. Current type: {type(other)}"

    def copy(self):
        from copy import copy
        return copy(self)
        if isinstance(other, int) and other > 0:
            from copy import copy
            res = []
            for i in range(other):
                res.append(copy(self))
            return res

    @property
    def tokens(self) -> List[str]:
        """Getter of tokens variable

        Returns:
            List[str]: List of tokens of the sentence
        """
        return self._tokens

    @tokens.setter
    def tokens(self,tokens : List[str]) -> None:
        """Setter of tokens variable. Must be a list of str

        Args:
            value (List[str]): list of token of the sentence
        """
        if not isinstance(tokens, type(None)):
            assert isinstance(tokens, list), f"token must be a list. Current type: {type(tokens)}"
            assert all(isinstance(tok, str) for tok in tokens), f"token must be a list of str. Current type: {[type(tok) for tok in tokens]}"
        self._tokens = copy.copy(tokens)

    @property
    def identifiant(self) -> int:
        """Getter of identifiant variable.

        Returns:
            int: unique identifiant of the sentence.
        """
        return self._identifiant

    @identifiant.setter
    def identifiant(self, identifiant: int) -> None:
        """Setter of identifiant variable.
        -1 symbolize a dummy identifiant.

        Args:
            identifiant (int): unique identifiant of the sentence (>=0 or ==-1).
        """
        if not isinstance(identifiant, type(None)):
            assert isinstance(identifiant, int), f"identifiant must be an int. Current type | Current value: {type(identifiant)} | {identifiant }"
        self._identifiant = identifiant

    @staticmethod
    def len(self) -> int:
        return len(self.tokens)

    @staticmethod
    def list_suppr_pad(tokens, padding_mark="<pad>", strict=False)-> List[int]:
        """retourne la liste des index du padding à supprimer par ordre décroissant.

        Args:
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to False.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        Example
        >>> Snt.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="<pad>", strict=False)
        [6, 2, 1]
        >>> Snt.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="<pad>", strict=True)
        [6, 2, 1, 0]
        >>> Snt.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="ci", strict=True)
        [4]
        >>> Snt.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="padding", strict=True)
        []
        """
        stop = -1 if strict else 0
        list_suppr_pad = []
        for i in range(len(tokens)-1, stop, -1):
            if tokens[i] == padding_mark:
                list_suppr_pad.append(i)
        return list_suppr_pad

    @staticmethod
    def list_fusion_bpe(tokens: List[str], BPE_mark: str = '@@') -> List[int] :
        """retourne la liste décroissante des tokens contenant une marque de BPE à la fin

        Returns:
            List[int]: liste décroissante des tokens contenant une marque de BPE à la fin
        >>> Snt.list_fusion_bpe(tokens= ["Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        [[6, 5], [1, 0]]
        >>> Snt.list_fusion_bpe(tokens= ["lu@@", "bu@@", "lu@@", "le", ".", "<eos>"])
        [[3, 2, 1, 0]]
        >>> Snt.list_fusion_bpe(tokens= ["Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], BPE_mark="bpe_mark")
        """
        assert isinstance(tokens, list), f"tokens must be a list. Current type: {type(tokens)}"
        assert all(isinstance(tok, str) for tok in tokens), f"tokens must be a list of str. Current type: {[type(tok) for tok in tokens]}"
        assert isinstance(BPE_mark, str), f"BPE_mark must be a str. Current type: {type(BPE_mark)}"
        assert tokens, "Liste vide"
        assert not tokens[-1].endswith(BPE_mark), f"Dernier token contenant une marque de BPE. Sentence: {tokens}"
        from Utils import Utils

        liste_bpe = []
        flag = False
        for i in range(len(tokens)-1, -2, -1):
            if tokens[i].endswith(BPE_mark):
                flag = True
                liste_bpe.append(i+1)
            elif flag:
                liste_bpe.append(i+1)
                flag = False
        return Utils.regrouper_indices_consecutifs(liste_bpe) if len(liste_bpe) >= 1 else None
        # return [ i for i in range(len(tokens) -1, -1, -1) if tokens[i].endswith(BPE_mark) ]

    def append(self, value: str):
        """Ajoute un (ou plusieurs via liste) token(s) à la fin de la phrase

        Args:
            value (str or List[str]): token ou liste à ajouter à la fin de la phrase
        
        Tests:
        >>> s1 = Snt(identifiant= 3, tokens= ['Ce@@', 'ci', 'est', '<pad>', 'un'])
        >>> s1.append('test')
        >>> print(s1)
        {'_identifiant': 3, '_tokens': ['Ce@@', 'ci', 'est', '<pad>', 'un', 'test']}
        >>> s1 = Snt(identifiant= 3, tokens= ['Ce@@', 'ci', 'est', 'test', '<pad>', 'un'])
        >>> s1.append(['test1', 'test2'])
        >>> print(s1)
        {'_identifiant': 3, '_tokens': ['Ce@@', 'ci', 'est', 'test', '<pad>', 'un', 'test1', 'test2']}
        """
        assert isinstance(value, str) or (isinstance(value, list) and all([isinstance(val, str) for val in value])), f"[DEBUG] Snt().append only supports strings and List[string]. Current type : {type(value)}"
        if isinstance(value, str):
            self.tokens.append(value)
        elif isinstance(value, List):
            self.tokens += value
    
    def insert(self, index: int, value) -> None:
        """Insert value en position index dans la liste de tokens

        Args:
            index (int): position de l'index où inserer value
            value (str): token unique à insérer
        
        Tests:
        >>> s1 = Snt(identifiant= 3, tokens= ["Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> s1.insert(0, "test")
        >>> print(s1)
        {'_identifiant': 3, '_tokens': ['test', 'Ce@@', 'ci', 'est', '<pad>', 'un', 'te@@', 'st', '.', '<eos>']}
        """
        assert isinstance(index, int), f"index must be an int. Current type: {type(index)}"
        assert isinstance(value, str), f"value must be a str. Current type: {type(value)}"
        self.tokens.insert(index, value)

    def toJSON(self):
        import json
        return json.dumps(
            self,
            default=lambda o: o.__dict__,
            sort_keys=True,
            indent=4)

    def suppr_pad(self, list_index: List[int] = None, padding_mark='<pad>', strict=False) -> List[int]:
        """Supprime le padding de la phrase et retourne une liste contenant les index supprimés

        Args:
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to False.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        
        Examples:
        >>> s1 = Snt(identifiant= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> list_index = Snt.list_suppr_pad(s1.tokens, padding_mark="<pad>", strict=False)
        >>> s1.suppr_pad(list_index)
        [6, 2, 1]
        >>> s2 = Snt(identifiant= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
        >>> s2.suppr_pad(strict=True)
        [6, 2, 1, 0]
        """
        list_index = Snt.list_suppr_pad(self.tokens, padding_mark=padding_mark, strict=strict) if list_index is None else list_index

        for i in list_index:
            del self.tokens[i]
        return list_index

    def fusion_bpe(self, list_bpe: List[int] = None, BPE_mark: str = '@@') -> List[int]:
        """Fusionne les tokens BPEisés

        Args:
            list_bpe (List[int]): Liste des indexs décroissant des tokens contenant une marque de BPE.
        
        Returns:
            None

        Example:
        >>> snt3 = Snt(2, tokens=["lu@@", "bu@@", "lu@@", "le", ".", "<eos>"])
        >>> snt3.fusion_bpe()
        [[3, 2, 1, 0]]
        >>> print(snt3)
        {'_identifiant': 2, '_tokens': ['lubulule', '.', '<eos>']}
        """
        groupes_bpe = Snt.list_fusion_bpe(self.tokens, BPE_mark=BPE_mark) if list_bpe is None else list_bpe
        
        if groupes_bpe is not None:
            # Cas particulier où il n'y a pas de bpe dans la phrase
            for groupe in groupes_bpe:
                if len(groupe) >= 1:
                    for i in range(groupe[1], groupe[-1] -1 , -1):
                        self.tokens[i] = f"{self.tokens[i].split(BPE_mark)[0]}{self.tokens[i+1]}"
                        del self.tokens[i+1]

        return groupes_bpe
    
    def test_(self, _id = 0):
        print(f"[DEBUG] Production d'une Snt de Test")
        self.identifiant = _id
        self.tokens = [f"{_id}", "Pro", "Duc", "tion", "d'", "une", "Snt", "de", "test", "."]


if __name__ == "__main__":
    import doctest
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    doctest.testmod()
    # print()

    # print(Snt.list_suppr_pad(["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"], padding_mark="<pad>", strict=True))

    # s2 = Snt(identifiant= 3, tokens= ["<pad>", "<pad>", "<pad>", "Ce@@", "ci", "est", "<pad>", "un", "te@@", "st", ".", "<eos>"])
    # print(s2.suppr_pad(strict=True))
    # s1 = Snt(2, tokens=["lu@@", "bu@@", "lu@@", "le", ".", "<eos>"])
    # liste = Snt.list_fusion_bpe(s1.tokens)
    # s1.fusion_bpe(liste)
    # print(s1.tokens)


