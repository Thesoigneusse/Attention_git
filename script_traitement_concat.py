import torch
from Snt import Snt
from Matrice import Matrice
import Utils_data
import Utils_concat

for id in range(2308):
    print(f"[DEBUG] id: {id}")
    r_path=f"/home/getalp/lopezfab/lig/temp/temp/test_attn/{id}.json"
    data=Utils_data.lecture_data(r_path)
    _OUTPUT_PATH=f"/home/getalp/lopezfab/Documents/concat/{id}"

    # Traitement des Phrases extraites
    ssl = data['src_segments_labels']
    src = Snt(identifiant=-1, tokens=Utils_concat.ajoute_eos_tokens_src(_snt=data["src_tokens"].split(), src_segments_labels=ssl))
    src_cutted = Utils_concat.full_sentence_to_ctx_and_crt(src)
    for i in range(len(src_cutted)-1, -1, -1):
        if src_cutted[i] == ["<eos>"]:
            del src_cutted[i]
    if len(src_cutted) > 1:
        # Extraction des phrases de contexte et de la phrase courante
        ctxs = []
        for k in range(len(src_cutted[:-1])):
            ctxs.append(Snt(identifiant=int(data["id"]) - len(src_cutted[k:-1]), tokens = src_cutted[k]))
        full_ctx = Snt(identifiant= id - len(ctxs), tokens= ctxs[0].tokens)
        # print(f"[debug] len(full_ctx): {len(full_ctx)}")
        if len(ctxs) > 1:
            for snt in ctxs[1:]:
                full_ctx.tokens += snt.tokens
                # print(f"[debug] len(full_ctx): {len(full_ctx)}")
        crt = Snt(identifiant=data['id'], tokens= src_cutted[-1])
        full_ctx.tokens += crt.tokens

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
        # print((f"[DEBUG] taille des matrices: {layers[0][0].matrice.size()}"))
        # print(f"[DEBUG] taille des phrases complète vs tailles respectives: {sum([len(ctx) for ctx in ctxs] + [len(crt)])} vs. [{[len(ctx) for ctx in ctxs]}, {len(crt)}]")

        # Pour chaque layer, pour chaque tête d'attention, on découpe la matrice en combinaison de k3*k3, k3*k2... k2*k3, k2*k2,... crt*crt
        for layer in range(len(layers)):
            for head in range(len(layers[layer])):
                full_matrice_cutted = Utils_concat.cut_matrix_into_sentences(layers[layer][head], src_cutted)
                # Dernière liste correspond à la phrase courante vers les phrases de contexte et la phrase courante
                layers[layer][head].matrice = torch.cat([matrice.matrice for matrice in full_matrice_cutted[-1][:-1]], dim = 1) # Remplacer [:] par [:-1] pour n'avoir que le contexte
        # print(f"[DEBUG] taille de la matrice découpée: {[matrice.matrice.size() for matrice in full_matrice_cutted[-1]]}")
        # print(f"[DEBUG] taille de la matrice reconstituée: {layers[layer][head].matrice.size()}")
        # print(f"[DEBUG] élément matrice: {layers[0][0].matrice[0:2, 3:9]}")
        for layer in range(len(layers)):
            for head in range(len(layers[layer])):
                # Traitement des matrices (suppression inf uniforme et normalisation)
                layers[layer][head].suppr_inf()
                layers[layer][head].norm_tensor()
                layers[layer][head].ecriture_xslx(crt=crt, ctx=full_ctx, precision= 4, absolute_folder=f"{_OUTPUT_PATH}/full_matrice/{layer}", filename=f"{head}", create_folder_path=True)