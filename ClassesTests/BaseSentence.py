from dataclasses import dataclass, field
from typing import List
import sys
sys.path.append('/home/getalp/lopezfab/Bureau/Attention_git')

from BaseWord import BaseWord

@dataclass
class BaseSentence():
    identifiant: int
    tokens: List[BaseWord] = field(default_factory=list)

    def __len__(self) -> int:
        """Retourne la longueur de la phrase courante en nombre de tokens.

        Returns:
            int: nombre de tokens dans la phrase courante
        """
        return len(self.tokens)
    
    def __add__(self, other: 'BaseSentence') -> 'BaseSentence':
        """Additionne 2 phrases en concaténant les tokens.

        Args:
            other (BaseSentence): BaseSentence à concaténer à la suite de la BaseSentence courante

        Returns:
            BaseSentence: New instance of BaseSentence with the concatenated tokens
        """
        raise NotImplementedError

    def copy(self) -> 'BaseSentence':
        """ Copie la BaseSentence courante
        """
        raise NotImplementedError
    
    @staticmethod
    def list_suppr_pad(tokens: List[str], padding_mark="<pad>", strict=True)-> List[int]:
        """retourne la liste des index du padding à supprimer par ordre décroissant.

        Args:
            tokens (List[str]): Liste de tokens
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to False.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        
        Tests:
            >>> s2 = BaseSentence(identifiant=-1,  tokens=[BaseWord(identifiant=0, token="<pad>"), BaseWord(identifiant=1, token="<pad>"), BaseWord(identifiant=2, token="<pad>"), BaseWord(identifiant=3, token="Ce@@"), BaseWord(identifiant=4, token="ci"), BaseWord(identifiant=5, token="est"), BaseWord(identifiant=6, token="<pad>"), BaseWord(identifiant=7, token="un"), BaseWord(identifiant=8, token="te@@"), BaseWord(identifiant=9, token="st"), BaseWord(identifiant=10, token="."),  BaseWord(identifiant=11, token="<eos>")])
            >>> print(BaseSentence.list_suppr_pad(s2.tokens, strict=False))
            [6, 2, 1]
            >>> print(BaseSentence.list_suppr_pad(s2.tokens))
            [6, 2, 1, 0]
            >>> print(BaseSentence.list_suppr_pad(s2.tokens, padding_mark="ci", strict=True))
            [4]
            >>> print(BaseSentence.list_suppr_pad(s2.tokens, padding_mark="padding", strict=True))
            []
        """
        stop = -1 if strict else 0
        list_suppr_pad = []
        for i in range(len(tokens)-1, stop, -1):
            if tokens[i].is_padding_mark(padding_mark):
                list_suppr_pad.append(i)
        return list_suppr_pad

    def suppression_padding(self, list_index: List[int] = None, padding_mark='<pad>', strict=True) -> List[int]:
        """Supprime le padding de la phrase et retourne une liste contenant les index supprimés

        Args:
            list_index (List[int], optional): Liste décroissante d'index à supprimer. Defaults to None.
            padding_mark (str, optional): Chaîne de caractères correspondant au token de padding. Defaults to "<pad>".
            strict (bool, optional): Permet de garder un token de padding (strict= False) ou non (strict = True). Defaults to False.

        Returns:
            List[int]: Liste d'index dont la position est à supprimer
        """
        if list_index is None:
            # Si la liste des index  correspondant aux tokens de padding n'est pas fournie, on la génère
            list_index = BaseSentence.list_suppr_pad(self.tokens, padding_mark=padding_mark, strict=strict)

        # On supprime les tokens de padding
        for i in list_index:
            del self.tokens[i]
        return list_index

    @staticmethod
    def list_merge_bpe(sentence: 'BaseSentence', separator: str = "@@")-> List[int]:
        """retourne la liste des index des tokens BPEs à supprimer par ordre décroissant.

        Args:
            separator (str, optional): Séparateur des tokens BPEs. Defaults to "@@".

        Returns:
            List[int]: Liste d'index dont la position est à supprimer

        Tests:
        >>> s1 = BaseSentence.load_BaseSentence(nb_BaseWord = 5, identifiant=1)
        >>> s1.tokens[3] += BaseWord(identifiant=-1, token="@@")
        >>> BaseSentence.list_merge_bpe(s1, separator="@@")
        [[4, 3]]
        >>> s2 = BaseSentence(identifiant=-1,  tokens=[BaseWord(identifiant=0, token="<pad>"), BaseWord(identifiant=1, token="<pad>"), BaseWord(identifiant=2, token="<pad>"), BaseWord(identifiant=3, token="Ce@@"), BaseWord(identifiant=4, token="ci"), BaseWord(identifiant=5, token="est"), BaseWord(identifiant=6, token="<pad>"), BaseWord(identifiant=7, token="un"), BaseWord(identifiant=8, token="te@@"), BaseWord(identifiant=9, token="st"), BaseWord(identifiant=10, token="."), BaseWord(identifiant=11, token="<eos>")])
        >>> BaseSentence.list_merge_bpe(s2, separator="@@")
        [[9, 8], [4, 3]]
        """
        from icecream import ic

        # Liste des groupes de BPEs à merge
        list_merge_bpe = []
        # Indique si le token précédant était aussi un BPE
        is_previous_tok_bpe = False

        # On parcours les index des tokens dans le sens décroissant
        # Pour chaque token
        for i in range(len(sentence.tokens)-1, 0, -1):
            if sentence.tokens[i-1].is_bpe(separator):
                if not is_previous_tok_bpe:
                    # On ajoute un groupe d'index contenant l'index du token courant à la liste des tokens à merge
                    list_merge_bpe.append([i])
                    is_previous_tok_bpe = True
                else:
                    # On ajoute l'index du token courant au groupe précédant
                    list_merge_bpe[-1].append(i)
            else:
                # Si le token ne se termine pas par le séparateur et que le token précédant n'est pas un BPE
                if is_previous_tok_bpe:
                    # On ajoute l'index du token courant au groupe précédant
                    # et on indique que le token précédant n'est pas un BPE
                    list_merge_bpe[-1].append(i)
                    is_previous_tok_bpe = False
                else:
                    # On indique que le token précédant n'est pas un BPE
                    is_previous_tok_bpe = False

        return list_merge_bpe

    def merge_bpe(self, list_groupes_index: List[List[int]] = None, separator: str = "@@") -> List[List[int]]:
        """Merge les tokens BPEs et retourne une liste contenant les index supprimés

        Args:
            list_index (List[List[int]]): Liste décroissante de groupe d'index à supprimer.
            separator (str, optional): Séparateur des tokens BPEs. Defaults to "@@".

        Returns:
            List[List[int]]: Liste de groupe d'index dont la position est à supprimer

        Tests:
        >>> s1 = BaseSentence(identifiant=1, tokens=[BaseWord(identifiant=0, token='0'), BaseWord(identifiant=1, token='1'), BaseWord(identifiant=2, token='2'), BaseWord(identifiant=3, token='3@@'), BaseWord(identifiant=4, token='4')])
        >>> s1.merge_bpe()
        [[4, 3]]
        >>> print(s1)
        BaseSentence(identifiant=1, tokens=[BaseWord(identifiant=0, token='0'), BaseWord(identifiant=1, token='1'), BaseWord(identifiant=2, token='2'), BaseWord(identifiant=4, token='34')])
        """

        # Si on n'a pas les index, on les calcules
        if list_groupes_index is None:
            list_groupes_index = BaseSentence.list_merge_bpe(self, separator=separator)
        
        # Pour chaque groupe d'index à fusionner
        for groupe in list_groupes_index:
            # On parcours les tokens du groupe à partir du 2ème token
            for i in groupe[1:]:
                # On fusionne le token courant au premier token du groupe et on supprime le token courant
                self.tokens[groupe[0]].token = self.tokens[i].token[:-len(separator)] + self.tokens[groupe[0]].token
                del self.tokens[i]

        return list_groupes_index

    @staticmethod
    def load_BaseSentence(nb_BaseWord: int, identifiant = -1) -> 'BaseSentence':
        """Charge une BaseSentence de taille nb_BaseWord.

        Args:
            nb_BaseWord (int): Nombre de BaseWord de la BaseSentence.

        Returns:
            BaseSentence: BaseSentence de taille nb_BaseWord.
        
        Tests:
        >>> BaseSentence.load_BaseSentence(5, identifiant=1)
        BaseSentence(identifiant=1, tokens=[BaseWord(identifiant=0, token='0'), BaseWord(identifiant=1, token='1'), BaseWord(identifiant=2, token='2'), BaseWord(identifiant=3, token='3'), BaseWord(identifiant=4, token='4')])
        """
        return BaseSentence(identifiant= identifiant,
                            tokens= [BaseWord(identifiant=i, token=str(i)) for i in range(nb_BaseWord)])

        
def main():
    from icecream import ic
    print("BaseSentence main function")

    s2 = BaseSentence(identifiant=-1,  tokens=[BaseWord(identifiant=0, token="<pad>"),
                                               BaseWord(identifiant=1, token="<pad>"),
                                               BaseWord(identifiant=2, token="<pad>"),
                                               BaseWord(identifiant=3, token="Ce@@"),
                                               BaseWord(identifiant=4, token="ci"),
                                               BaseWord(identifiant=5, token="est"),
                                               BaseWord(identifiant=6, token="<pad>"),
                                               BaseWord(identifiant=7, token="un"),
                                               BaseWord(identifiant=8, token="te@@"),
                                               BaseWord(identifiant=9, token="st"),
                                               BaseWord(identifiant=10, token="."), 
                                               BaseWord(identifiant=11, token="<eos>")])
    ic(BaseSentence.list_suppr_pad(s2.tokens))
    ic(BaseSentence.list_suppr_pad(s2.tokens, strict=True))
    ic(s2)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    print(f"[DEBUG]Doctest passed for BaseSentence")


    main()