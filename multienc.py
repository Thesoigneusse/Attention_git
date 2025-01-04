import Utils_data as ud
import Utils
import torch
import Matrice
import Sl_matrice
from Snt import Snt

precision = 8


for id in range(2308):
    print(f"sentence : {id}")
    precision = 8
    r_path=f"/home/getalp/lopezfab/lig/temp/temp/temp/han_attn2/{id}.json"
    OUTPUT_PATH = f"/home/getalp/lopezfab/Documents/{id}"

    # Lecture des données
    data=ud.lecture_data(r_path)
    crt, ctxs, ctxs_heads, sl_heads = ud.lecture_multi_enc_objet(data)

    # Corrections des données dans les cas où il y a moins de 3 contextes
    mask = torch.ones(sl_heads[0].matrice.shape[1], dtype = torch.bool)
    for k in range(len(ctxs)-1, -1, -1):
        # On supprime les contextes inutiles
        if len(ctxs[k].tokens) == 1 or (len(ctxs[k].tokens) > 1 and ctxs[k].tokens[-2] == "<pad>"):
            del ctxs[k]
            del ctxs_heads[k]
            mask[k] = False
        else:
            # On corrige un problème de padding qui apparait quand il y a moins de 3 contextes
            for h in range(len(ctxs_heads[0])):
                ctxs_heads[k][h].matrice = ctxs_heads[k][h].matrice[..., -len(ctxs[k]):]

    # S'il y a au moins une phrase de contexte on l'a traite
    if len(ctxs) >= 1:
        for h in range(len(sl_heads)):
            sl_heads[h].matrice = sl_heads[h].matrice[:, mask]
        
        # Process de la phrase courante
        list_crt_suppr_pad = crt.suppr_pad()
        list_crt_fusion_bpe = crt.fusion_bpe()

        mean_ctxs_heads = []
        # Traitement de chaque contexte
        for k in range(len(ctxs)):
            # Process de la phrases de contexte K
            list_ctx_suppr_pad = ctxs[k].suppr_pad()
            list_ctx_fusion_bpe = ctxs[k].fusion_bpe()

            for head in range(len(ctxs_heads[k])):
                # Traitement de chaque tête d'attention entre la phrase courante et chaque phrase de contexte k
                ctxs_heads[k][head].suppr_pad(row_list_suppr_pad= list_crt_suppr_pad, col_list_suppr_pad=  list_ctx_suppr_pad)
                ctxs_heads[k][head].fusion_bpe(row_list_fusion_bpe= list_crt_fusion_bpe, col_list_fusion_bpe= list_ctx_fusion_bpe)
                ctxs_heads[k][head].suppr_inf()

            # Pour chaque phrase de contexte, on récupère la moyenne des poids d'attention 
            # de la phrase courante vers la phrase de contexte (doit être effectuée avant la normalisation )
            mean_ctxs_heads.append(Matrice.Matrice(Utils.mean_matrices([ctxs_heads[k][head].matrice for head in range(len(ctxs_heads[k])) ])))
            mean_ctxs_heads[k].norm_tensor() # On peut normaliser la moyenne car on ne l'utilise pas dans la contextualisation

        for sl_head in range(len(sl_heads)):
            # Traitement de chaque tête d'attention entre la phrase courante et l'ensemble des phrases de contexte
            sl_heads[sl_head].suppr_pad(row_list_suppr_pad= list_crt_suppr_pad)
            sl_heads[sl_head].fusion_bpe(row_list_fusion_bpe= list_crt_fusion_bpe, action= "mean")
            

        # On récupère la moyenne des têtes d'attention du mécanisme sentence level
        mean_sl_heads = Sl_matrice.Sl_matrice(Utils.mean_matrices([sl_heads[sl_head].matrice for sl_head in range(len(sl_heads))]))
        mean_sl_heads.norm_tensor() # On peut normaliser la moyenne car on l'utilise pas dans la contextualisation

        
        # Contextualisation entre le mécanisme d'attention word-level et le mécanisme d'attention sentence-level
        # Récupération de l'ensemble des phrases de contexte en une seule d'identifiant -1
        full_ctx = Snt(identifiant=-1, tokens=ctxs[0]) if len(ctxs)>= 1 else None
        if full_ctx is not None:
            for k in range(1, len(ctxs)):
                full_ctx += ctxs[k]

        # écriture Token-level K x H x crt x ctxs[k] + means
        for k in range(len(ctxs)):
            # Pour chaque phrase de contexte K, 
            # On écrit les matrices de chaque heads entre la phrase  courante et la phrase de contexte K
            # Puis la moyenne de têtes
            for head in range(len(ctxs_heads[k])):
                ctxs_heads[k][head].ecriture_xslx(crt= crt, 
                                                    ctx= ctxs[k],
                                                    absolute_folder= f"{OUTPUT_PATH}/token_level/{head}", 
                                                    filename=f"ctx_{k}", 
                                                    precision=precision, 
                                                    create_folder_path=True)
            mean_ctxs_heads[k].ecriture_xslx(crt= crt,
                                                ctx= ctxs[k],
                                                absolute_folder= f"{OUTPUT_PATH}/token_level", 
                                                filename=f"mean_ctx_{k}", 
                                                create_folder_path=True)

        # écriture Sentence-level H x crt x nb_ctx + means
        for sl_head in range(len(sl_heads)):
            # Pour chaque tête d'attention sentence-level,
            # On écrit la tête d'attention entre la phrase courante et les K phrases
            # Puis on écrit la moyenne des têtes d'attention
            # TODO: à vérifier si sl_heads est en mode k3 x k2 x k1 ou k1 x k2 x k3
            # Si sl_heads est en mode  k1 x k2 x k3 alors .flip le passe en mode k3 x k2 x k1 pour plus de lisibilité
            sl_heads[sl_head].matrice=sl_heads[sl_head].matrice.flip(1)
            sl_heads[sl_head].ecriture_xslx(crt= crt, 
                                                absolute_folder= f"{OUTPUT_PATH}/sentence_level", 
                                                filename=f"sl_head_{sl_head}", 
                                                precision=precision, 
                                                create_folder_path=True)
        mean_sl_heads.ecriture_xslx(crt= crt,
                                        absolute_folder= f"{OUTPUT_PATH}/sentence_level", 
                                        filename=f"mean", 
                                        precision=precision, 
                                        create_folder_path=True)


        # Traitement et écriture des phrases contextualisé crt x nb_ctx
        for h_sl in range(len(sl_heads)):
            for h_tl in range(len(ctxs_heads[0])):
                
                temp = sl_heads[h_sl].contextualise_matrice([ctxs_heads[k][h_tl] for k in range(sl_heads[h_sl].matrice.size(1))])
                temp.ecriture_xslx(crt= crt,
                                            ctx= full_ctx,
                                            absolute_folder= f"{OUTPUT_PATH}/full_matrice/sl_{h_sl}", 
                                            filename=f"head_{h_tl}", 
                                            precision=8,
                                            create_folder_path=True)


## OUTPUT
# OUTPUT_PATH/{id}/
#  - sentence_level
#     - sl_heads_{head}.xslx
#        ...
#  - token_level
#     - {head}
#        - ctx_{k}.xslx
#           ...
#  - full_matrice
#     - sl_{sl_head}
#        - head_{head}.xslt
#           ...
#        ...