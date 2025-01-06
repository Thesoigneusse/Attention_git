import torch
from Snt import Snt
from Matrice import Matrice
import Utils_data
import Utils_concat
import importlib
importlib.reload(Utils_data)
importlib.reload(Utils_concat)
_FULL_SNT = False # permet prendre en compte la phrase courante ou non
_ON_SERV = True

_PATH = f"/home/getalp/lopezfab" if _ON_SERV else f"/home/getalp/lopezfab/lig"
for id in range(187, 2308):
    print(f"[DEBUG] id: {id}")
    r_path=f"{_PATH}/temp/temp/test_attn/{id}.json"
    data=Utils_data.lecture_data(r_path)
    _OUTPUT_PATH=f"{_PATH}/temp_phd/concat/{id}"

    # Traitement des Phrases extraites
    ssl = data['src_segments_labels']
    src = Snt(identifiant=-1, tokens=Utils_concat.ajoute_eos_tokens_src(_snt=data["src_tokens"].split(), src_segments_labels=ssl))
    src_cutted = Utils_concat.full_sentence_to_ctx_and_crt(src)

    # Au moins une phrase de contexte et la phrase courante (+ 1 car contient une phrase vide quand le nb de contexte est inférieur à la normale)
    if len(src_cutted) > 2: 
        # Extraction des phrases de contexte et de la phrase courante
        ctxs = []
        for k in range(len(src_cutted[:-1])):
            ctxs.append(Snt(identifiant=int(data["id"]) - len(src_cutted[k:-1]), tokens = src_cutted[k]))
        # print(f"[debug] len(full_ctx): {len(full_ctx)}")

        # Extraction des différentes matrices à travers les 6 layers et les 8 têtes d'attention de chaque layer
        layers = []
        for layer in range(len(data['heads_enc_attn'])): # Pour chaque layer
            heads = []
            for head in range(len(data['heads_enc_attn'][layer])): # on extrait chaque tête par layer
                full_matrice = torch.tensor(data['heads_enc_attn'][layer][head])
                full_matrice = full_matrice.squeeze() # on supprime une dimension qui semble inutile (=1)
                heads.append(Matrice(full_matrice))
            layers.append(heads)
        # layers : L x [nb_heads x [torch.Tensor(N x N)]]

        # Traitement du cas particulier où un contexte n'est pas présent. Suppression des reliquats dans le contexte et les matrices d'attention
        for i in range(len(ctxs)-1, -1, -1):
            if ctxs[i].tokens == ["<eos>"]:
                del ctxs[i]
                del src_cutted[i]
                for layer in range(len(layers)):
                    for head in range(len(layers[layer])):
                        layers[layer][head].matrice = torch.cat([layers[layer][head].matrice[1:, 1:]])

        # Récupération de l'ensemble des tokens
        full_ctx = Snt(identifiant= id - len(ctxs), tokens= ctxs[0].tokens)
        if len(ctxs) > 1:
            for snt in ctxs[1:]:
                full_ctx.tokens += snt.tokens
                # print(f"[debug] len(full_ctx): {len(full_ctx)}")
        crt = Snt(identifiant=data['id'], tokens= src_cutted[-1])
        if _FULL_SNT: # permet prendre en compte la phrase courante ou non
            full_ctx.tokens += crt.tokens

        # print((f"[DEBUG] taille des matrices: {layers[0][0].matrice.size()}"))
        # print(f"[DEBUG] taille des phrases complète vs tailles respectives: {sum([len(ctx) for ctx in ctxs] + [len(crt)])} vs. [{[len(ctx) for ctx in ctxs]}, {len(crt)}]")

        # Pour chaque layer, pour chaque tête d'attention, on découpe la matrice en combinaison de k3*k3, k3*k2... k2*k3, k2*k2,... crt*crt
        for layer in range(len(layers)):
            for head in range(len(layers[layer])):
                full_matrice_cutted = Utils_concat.cut_matrix_into_sentences(layers[layer][head], src_cutted)
                # Dernière liste correspond à la phrase courante vers les phrases de contexte et la phrase courante
                # _FULL_SNT permet prendre en compte la phrase courante ou non
                if _FULL_SNT:
                    layers[layer][head].matrice = torch.cat([matrice.matrice for matrice in full_matrice_cutted[-1][:]], dim = 1)
                else:
                    layers[layer][head].matrice = torch.cat([matrice.matrice for matrice in full_matrice_cutted[-1][:-1]], dim = 1)
        # print(f"[DEBUG] taille de la matrice découpée: {[matrice.matrice.size() for matrice in full_matrice_cutted[-1]]}")
        # print(f"[DEBUG] taille de la matrice reconstituée: {layers[layer][head].matrice.size()}")
        # print(f"[DEBUG] élément matrice: {layers[0][0].matrice[-2, ...]}")
        for layer in range(len(layers)):
            for head in range(len(layers[layer])):
                layers[layer][head].suppr_inf()
                layers[layer][head].norm_tensor()
                layers[layer][head].ecriture_xslx(crt=crt, ctx=full_ctx, precision= 4, absolute_folder=f"{_OUTPUT_PATH}/full_matrice/{layer}", filename=f"{head}", create_folder_path=True)
