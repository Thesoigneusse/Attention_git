from typing import TypedDict

class Coref(TypedDict):
        id: str
        fileid: str
        span: str
        coref_class: str
        mention: str
