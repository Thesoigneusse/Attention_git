# N : taille de la phrase courante
# k : nombre de phrase de contexte
# M_k : Taille de la phrase de contexte k
# M : taille des phrases de contexte
# S : Taille de la fusion des phrases de contexte et phrase courante
# S_k : Taille de la fusion des phrases de contexte et phrase courante découpées
# nb_heads : nombre de tête d'attention
# L : nombre de layers
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
from Utils import Utils_concat
from Utils import Utils_data
from Classes.Concat_matrice import Concat_matrice
import torch
torch.no_grad()


for id in range(2308):
    _SRC = True
    _TO_PROCESS = "crt_to_ctxs"
    print(f"[DEBUG] id: {id}")
    r_path=f"/home/getalp/lopezfab/temp/temp/test_attn/{id}.json"
    data=Utils_data.lecture_data(r_path)
    data = Utils_data.lecture_concat_objet(data)
    # print(f"[debug] data.keys: {data.keys()}")
    _OUTPUT_PATH=f"/home/getalp/lopezfab/Documents/concat/{id}" # TODO: +1 to suppr 

    # Traitement de la phrase src du modèle par concaténation
    if _SRC:
        src = Utils_concat.pre_traitement_src(identifiant = data['id'], 
                                                matrices = data['heads_enc_attn'],
                                                full_snt = data['src_tokens'])
        if src is not None:
            test = Concat_matrice(src['snt_cutted'][-1], src['snt_cutted'][:-1], src['layers'])
            test.fusion_bpe() # Suppression des bpes sur la full phrase

            crt_to_ctxs_crt = test.get_crt_to_ctxs() # On récupère uniquement la phrase courante vers les phrases de contexte. Taille: CRT x CTXS
            # crt_to_ctxs_crt = test.get_crt_to_ctxs_crt() # On récupère uniquement la phrase courante vers les phrases de contexte/crt. Taille: CRT x (CTXS + CRT)
            for l in range(len(crt_to_ctxs_crt)):
                for h in range(len(crt_to_ctxs_crt[l])):
                    crt_to_ctxs_crt[l][h].suppr_inf() # On supprime les valeurs inférieures à l'uniforme
                    crt_to_ctxs_crt[l][h].norm_tenseur() # On normalise les valeurs entre [0;1]
                    crt_to_ctxs_crt[l][h].ecriture_xslx(crt= test.crt, ctx = test.get_full_snt(to_process = _TO_PROCESS), absolute_folder=f"{_OUTPUT_PATH}/crt_to_full_ctxs_crt/{l}", filename= f'{h}', create_folder_path=True)

            # test.ecriture_xslx(data_to_write="crt_to_full_ctxs_crt", absolute_folder=_OUTPUT_PATH, create_folder_path=True)
        else:
            print(f"No src traitement on id: {id}")
    else:
        tgt = Utils_concat.pre_traitement_tgt(identifiant = data['id'],
                                            matrices = data['heads_dec_attn'],
                                            full_snt = data['tgt_tokens'])