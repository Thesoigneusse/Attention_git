# Une matrice context aware qui prend en entrée les différents éléments d'une matrice context-aware
from abc import abstractmethod
from Matrice import Matrice
from Snt import Snt
from typing import List
# N : taille de la phrase courante
# k : nombre de phrase de contexte
# M_k : Taille de la phrase de contexte k
# M : Taille de la fusion des phrases de contexte
# nb_heads : nombre de tête d'attention
# L : nombre de layers

class CA_matrice():
    def __init__(self, crt: Snt, ctxs: List[Snt], full_matrice: List[Matrice] = None):
        # Phrase courante
        # Dimension : N
        self.crt = crt

        # Phrase de contexte
        # Dimension : M
        self.ctxs = ctxs

        # Têtes d'attention entre la phrase courante et la phrase de contexte
        self.full_matrice = full_matrice

    @property
    def crt(self) -> Snt:
        """Retourne la phrase courante

        Returns:
            Snt: phrase courante
        """
        return self._crt
    @crt.setter
    def crt(self, value: Snt) -> None:
        assert isinstance(value, Snt), f"Sentence must be a Snt. Current Value: {type(value)}"
        self._crt = value

    @property
    def ctxs(self) -> List[Snt]:
        """Retourne la liste des phrases de contexte

        Returns:
            List[Snt]: Liste des phrases de contexte
        """
        return self._ctxs
    @ctxs.setter
    def ctxs(self, value: List[Snt]) -> None:
        assert isinstance(value, list), f"ctxs must be a list. Current Value: {type(value)}"
        assert all(isinstance(snt, Snt) for snt in value), f"ctxs must be a list of Snt. Current Value: {[type(snt) for snt in value]}"
        self._ctxs = value

    @property
    def full_matrice(self) -> List[Matrice]:
        """Retourne la liste des têtes d'attention entre la phrase courante et la phrase de contexte

        Returns:
            List[Matrice]: liste des têtes d'attention entre la phrase courante et la phrase de contexte
        """
        return self._heads
    @full_matrice.setter
    def full_matrice(self, value: List[Matrice]) -> None:
        if value is not None:
            assert isinstance(value, list), f"heads must be a list. Current Value: {type(value)}"
            assert all(isinstance(head, Matrice) for head in value), f"heads must be a list of Matrice. Current Value: {[type(head) for head in value]}"
        self._heads = value

    def __json__(self) -> str:
        return str(self.__dict__)

    def __repr__(self) -> str:
        return self.__json__()


    def full_ctx(self, identifiant= -1) -> Snt:
        """Retourne une instance Snt représentant l'entièreté des phrases de contextes en une phrase.
        Identifiant set à -1 par défaut 

        Args:
            identifiant (int, optional): Identifiant de la phrase. Defaults to -1.
        Returns:
            Snt: instance Snt représentant l'entièreté des phrases de contextes en une phrase
    
        """
        sentence = []
        for ctx in self.ctxs:
            sentence.extend(ctx.tokens)
        return Snt(identifiant=identifiant, tokens=sentence)

    def sentences_suppr_pad(self, strict=False, padding_mark='<pad>'):
        """Supprime le padding de la phrase courante et de la phrase de contexte.

        Args:
            padding_mark (str, optional): tokens de padding utilisé. Defaults to '<pad>'.

        Returns:
            list[int]: Liste des index correspondant à un token de padding dans la phrase courante.
            list[List[int]]: Liste des Listes des index correspondant à un token de padding dans la phrase pour chaque phrase de contexte.
        """
        list_crt_suppr_pad = self.crt.suppr_pad(padding_mark=padding_mark, strict=strict)
        list_ctxs_suppr_pad = []
        for k in range(len(self.ctxs)):
            list_ctxs_suppr_pad.append(self.ctxs[k].suppr_pad(padding_mark=padding_mark, strict=strict))
        return list_crt_suppr_pad, list_ctxs_suppr_pad

    def sentences_fusion_bpe(self, BPE_mark: str = '@@'):
        """Fusionne les BPEs dans la phrase ocurante et les phrases de contexte.

        Args:
            BPE_mark (str, optional): symbole utilisé pour indiquer un BPE. Defaults to '@@'.

        Returns:
            List[List[int]]: Liste des groupes de tokens correspondant à un seul et même mot dans la phrase courante.
            List[List[List[int]]]: Liste de Liste des groupes de tokens correspondant à un seul et même mot dans chaque phrase de contexte.
        """
        list_crt_fusion_bpe = self.crt.fusion_bpe(BPE_mark=BPE_mark)
        list_ctx_fusion_bpe = []
        for k in range(len(self.ctxs)):
            list_ctx_fusion_bpe.append(self.ctxs[k].fusion_bpe(BPE_mark=BPE_mark))
        return list_crt_fusion_bpe, list_ctx_fusion_bpe

    @abstractmethod
    def suppr_pad(self, BPE_mark: str = '@@'):
        raise NotImplementedError(f"Need to implement suppr_pad from CA_matrice")
        pass

    @abstractmethod
    def fusion_bpe(self, BPE_mark: str = '@@'):
        raise NotImplementedError(f"Need to implement fusion_bpe from CA_matrice")
        pass

    @abstractmethod
    def clean_matrice(self):
        raise NotImplementedError(f"Need to implement clean_matrice from CA_matrice")
        pass

    @abstractmethod
    def suppr_inf(self):
        raise NotImplementedError(f"Need to implement suppr_inf from CA_matrice")
        pass

    @abstractmethod
    def ecriture_xslx(self):
        raise NotImplementedError(f"Need to implement ecriture_xslx from CA_matrice")
        pass

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    import Utils_data

    id = 1850
    r_path=f"/home/getalp/lopezfab/lig/temp/temp/temp/han_attn2/{id}.json"
    data=Utils_data.lecture_data(r_path)
    crt, ctxs, ctxs_heads, sl_heads = Utils_data.lecture_objet(data)

    test = CA_matrice(crt= crt,
                        ctxs=ctxs,
                        full_matrice=sl_heads)
    print(test.sentences_suppr_pad())

    
    print(test.full_ctx())
