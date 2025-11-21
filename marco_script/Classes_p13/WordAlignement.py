from dataclasses import dataclass, asdict
from typing import Literal, Dict, Any, List

@dataclass
class WordAlignement:
    """
    Représente un alignement entre un mot de référence et un mot hypothèse.

    Attributes:
        index_reference (int): Index du mot dans la phrase de référence.
        index_hypothese (int): Index du mot dans la phrase hypothèse.
        alignement_type (Literal['del', 'ins', 'sub', 'match']): Type d'alignement.
    """

    alignement_type: Literal['del', 'ins', 'sub', 'match']
    index_reference: int
    index_hypothese: int | None

    def __post_init__(self):
        """Vérifie que le type d’alignement est correct."""
        valid_types = ['ins', 'del', 'sub', 'match']
        if self.alignement_type not in valid_types:
            raise ValueError(
                f"[DEBUG] alignement_type must be one of {valid_types}. "
                f"Current value: {self.alignement_type}"
            )
        elif self.alignement_type == "del":
            assert self.index_hypothese is None, ("[DEBUG] index_hypothese have to be None if alignement_type is 'del'."
                   f" Current self.alignement vs. self.index_hypothese: {self.alignement_type} vs. {self.index_hypothese}")
        else:
            assert self.index_hypothese is not None, ("[DEBUG index_hypothese have to be a value] if alignement_type is not 'del'."
                f" Current self.alignement vs. self.index_hypothese: {self.alignement_type} vs. {self.index_hypothese}")


    def to_Json(self) -> Dict[str, Any]:
        """
        Convertit l'objet en dictionnaire (pour JSON ou sérialisation).

        Returns:
            dict: Dictionnaire représentant l'alignement.
        """
        return {"__WordAlignement__": asdict(self)}

    @classmethod
    def from_Json(cls, data: Dict[str, Any]) -> 'WordAlignement':
        """
        Reconstruit un objet WordAlignement à partir d'un dictionnaire.

        Args:
            data (dict): Dictionnaire contenant les clés `index_reference`, `index_hypothese`, `alignement_type`.

        Returns:
            WordAlignement: Objet reconstruit.
        """
        return cls(**data)

    def to_list(self) -> List[Any]:
        """
        Retourne l'alignement sous forme de liste (ancien format Marco).

        Returns:
            list: [alignement_type, index_reference, index_hypothese]

        Examples:
            >>> WordAlignement("ins", 1, 2).to_list()
            ['ins', 1, 2]
        """
        return [self.alignement_type, self.index_reference, self.index_hypothese]


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    # Test rapide
    alignement = WordAlignement("ins", 1, 2)
    print("[debug]", alignement.to_Json())
    print("[debug]", alignement.to_list())
