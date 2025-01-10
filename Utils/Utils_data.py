import json
import torch
from typing import List
import copy
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from Classes.Snt import Snt
    from Classes.Matrice import Matrice
    from Classes.Sl_matrice import Sl_matrice

def lecture_data(absolute_file_path):
    """Read the data into the absolute_path file

    Args:
        absolute_path (str): absolute path to a .json file

    Returns:
        dict : dictionnary of the json file data
    """
    import mimetypes
    import json
    mime_type, encoding = mimetypes.guess_type(absolute_file_path)
    assert mime_type == "application/json", f"[DEBUG]File type error: need .json get {mime_type}"
    with open(f"{absolute_file_path}", "r") as f:
        data = json.load(f)
    return data

def ecriture_tsv(matrice: "Matrice", crt: "Snt", ctx: "Snt", absolute_folder:str, filename:str):
    """Écrit la matrice au format tsv à l'emplacement "{absolute_folder}/filename"

    Args:
        matrice (Matrice): Matrice des poids d'attention à écrire
        crt (Snt): Snt de la phrase courante
        ctx (Snt): Snt de la phrase de contexte
        absolute_folder (str): chemin absolu vers le dossier où écrire
        filename (str): nom du fichier dans lequel écrire la matrice
    """
    matrice_snts = mise_en_forme_matricielle(matrice, crt, ctx)
    with open(f"{absolute_folder}/{filename}.tsv", "w") as f:
        f.write("\n".join("\t".join(str(colonne) for colonne in ligne) for ligne in matrice_snts))

def mise_en_forme_matricielle(_matrice: "Matrice", _crt: "Snt", _ctx: "Snt", precision = 2) -> list:
    """Convertie une matrice ainsi que les phrases courante (de longueur n) et de contexte (de longueur m) en une matrice n x m au format:
    [[id_crt-id-ctx, ctx[0], ctx[1], ..., ctx[m]       ],
     [    crt[0]   , w_0_0 , w_1_0,  ...               ],
     ...................................................,
     ...................................................,
     [crt[len(crt)], w_n_1 , w_n_2, ... , w_n_m        ]]

    Args:
        _matrice (Matrice): Matrice des poids d'attention
        _crt (Snt): Snt de la phrase courante
        _ctx (Snt): Snt de la phrase de contexte

    Returns:
        list: liste de la phrase courante, de contexte et les poids d'attention
    """
    
    if _matrice is not None:
        matrice = copy.deepcopy(_matrice.matrice).tolist()
        crt = copy.deepcopy(_crt.tokens)
        ctx = copy.deepcopy(_ctx.tokens)
        assert len(matrice) == len(crt), f"[DEBUG]Matrice size error, matrice.size(): {matrice.size(dim= 0)} vs len(current sentence): {len(crt)}"
        assert len(matrice[0]) == len(ctx), f"[DEBUG]Matrice size error, matrice.size(): {matrice.size(dim= 1)} vs len(current sentence): {len(ctx)}"

        matrice.insert(0, ctx)
        matrice[0].insert(0, f"{_crt.identifiant}-{_ctx.identifiant}")
        for i in range(1, len(matrice)):
            matrice[i]= [round(x, precision) for x in matrice[i]]
            matrice[i].insert(0, crt[i - 1])
    return matrice

def check_path(absolute_folder: str, create_folder_path: bool):
    """Vérifie si le chemin existe. Le créé sinon.

    Args:
        absolute_folder (str): chemin absolu vers le dossier
        create_folder_path (bool): booléen pour créer ou non le dossier
    """
    import pathlib
    if not pathlib.Path(absolute_folder).exists() and create_folder_path:
        print(f"[DEBUG] Création du dossier : {absolute_folder}")
        pathlib.Path(absolute_folder).mkdir(parents=True)
    else:
        assert f"absolute_folder doit être un chemin existant or create_folder_path set to True. Current Path: {absolute_folder}"

def lecture_multi_enc_objet(data):
    """prends en entrée un json objet de la lecture de données d'un modèle multi_enc et retourne les éléments

    Args:
        data (str): json objet
    """
    # print(data.keys())
    from Classes.Snt import Snt
    # phrase courante
    crt = Snt(identifiant= int(data["id"]), tokens= data["crt"])
    
    # phrases de contexte
    ctxs = []
    ctxs_heads = []
    for k in range(len(data["ctxs"])):
        ctxs.append(Snt(identifiant= crt.identifiant - k - 1,tokens= data["ctxs"][k]))
        heads = []
        for h in range(len(data["heads"][0])):
            heads.append(Matrice(matrice = torch.DoubleTensor(data["heads"][k][h])))
        ctxs_heads.append(heads)
    correction_eos_context(ctxs)
    correction_eos_crt(crt)

    # sentence level heads
    sl_heads = []
    for h in range(len(data["SL_matrice"])):
        sl_heads.append(Sl_matrice(matrice = torch.DoubleTensor(data["SL_matrice"][h]).squeeze()))


    return (crt, ctxs, ctxs_heads, sl_heads)

def lecture_concat_objet(data):
    """prends en entrée un json objet de la lecture de données d'un modèle concat et retourne les éléments contenu

    Args:
        data (str): json objet
    """
    # print(data.keys())
    from Classes.Snt import Snt
    # phrase courante
    crt = Snt(identifiant= int(data["id"]), tokens= data["crt"])
    
    # phrases de contexte
    ctxs = []
    ctxs_heads = []
    for k in range(len(data["ctxs"])):
        ctxs.append(Snt(identifiant= crt.identifiant - k - 1,tokens= data["ctxs"][k]))
        heads = []
        for h in range(len(data["heads"][0])):
            heads.append(Matrice(matrice = torch.DoubleTensor(data["heads"][k][h])))
        ctxs_heads.append(heads)
    correction_eos_context(ctxs)
    correction_eos_crt(crt)

    # sentence level heads
    sl_heads = []
    for h in range(len(data["SL_matrice"])):
        sl_heads.append(Sl_matrice(matrice = torch.DoubleTensor(data["SL_matrice"][h]).squeeze()))


    return (crt, ctxs, ctxs_heads, sl_heads)


def correction_eos_context(ctxs: List["Snt"])-> None:
    for ctx in ctxs:
        ctx.tokens.append("<eos>")

def correction_eos_crt(crt: "Snt") -> None:
    crt.tokens.append("<eos>")