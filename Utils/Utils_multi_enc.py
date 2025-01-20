import torch
from typing import List
from typing import TYPE_CHECKING
if TYPE_CHECKING:
     from Classes.Matrice import Matrice
     from Classes.Sl_matrice import Sl_matrice
     from Classes.Multi_enc_matrice import Multi_enc_matrice
     from Classes.Snt import Snt


def pre_traitement_src(crt: "Snt", ctxs: List["Snt"], sl_heads: List["Sl_matrice"], ctxs_heads: List['Matrice']):
        from Classes.Multi_enc_matrice import Multi_enc_matrice
        # Corrections des données dans les cas où il y a moins de 3 contextes
        mask = torch.ones(sl_heads[0].shape[1], dtype = torch.bool)
        for k in range(len(ctxs)-1, -1, -1):
            # On supprime les contextes inutiles
            if len(ctxs[k].tokens) == 1 or (len(ctxs[k].tokens) > 1 and ctxs[k].tokens[-2] == "<pad>"):
                del ctxs[k]
                del ctxs_heads[k]
                mask[k] = False
            else:
                # On corrige un problème de padding qui apparait quand il y a moins de 3 contextes
                for h in range(len(ctxs_heads[0])):
                    ctxs_heads[k][h] = ctxs_heads[k][h][..., -len(ctxs[k]):]
        if len(ctxs) >= 1:
            for h in range(len(sl_heads)):
                sl_heads[h] = sl_heads[h][:, mask]

        return Multi_enc_matrice(crt= crt, ctxs = ctxs, ctxs_heads = ctxs_heads, sl_heads = sl_heads)



