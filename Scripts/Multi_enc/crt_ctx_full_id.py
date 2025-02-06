import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from Utils import Utils_data as ud
from Utils import Utils
import torch
with torch.no_grad():
    import Classes.Matrice as Matrice
    import Classes.Sl_matrice as Sl_matrice
    from Classes.Multi_enc_matrice import Multi_enc_matrice
    from Utils import Utils_multi_enc
    from Classes.Snt import Snt

    precision = 8


    for id in range(0, 2308):
        print(f"sentence : {id}")
        precision = 8
        r_path=f"/home/getalp/lopezfab/temp/temp/temp/han_attn2/{id}.json"
        OUTPUT_PATH = f"/home/getalp/lopezfab/Documents/multi_enc/{id}"

        # Lecture des données
        data=ud.lecture_data(r_path) 
        crt, ctxs, ctxs_heads, sl_heads = ud.lecture_multi_enc_objet(data)
    
        src = Utils_multi_enc.pre_traitement_src(crt, ctxs, sl_heads, ctxs_heads)
        # S'il y a au moins une phrase de contexte on l'a traite
        if len(src.ctxs) >= 1:

            # Process de la phrase courante
            src.suppr_pad()
            src.fusion_bpe()
            src.clean_matrice()
            test = src.get_crt_to_ctxs('full')
            # test List[List[Matrice]]. Taille : nb_sl_heads x nb_tl_heads x [crt x ctxs]
            for sl_heads in range(len(test)):
                for tl_heads in range(len(test[sl_heads])):
                    test[sl_heads][tl_heads].norm_tenseur()
                    test[sl_heads][tl_heads].ecriture_tsv(crt = src.crt, ctx = src.get_full_ctxs(), absolute_folder= f"{OUTPUT_PATH}/{sl_heads}", filename = f"{tl_heads}", create_folder_path=True)
