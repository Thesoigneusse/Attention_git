
import os
import sys
import torch

from Classes.EditDistance import EditDistance
from Classes.WordAlignement import WordAlignement
__ed_dict__ = {} # Dictionnaire de token produit par Marco

def str_edit_distance(str_ref, str_hyp, model=None, tokenizer=None):
    """Calcule la edit distanceentre du str str_ref (gold) avec le str str_hyp (prediction)

    Args:
        str_ref (str): str de la phrase de référence (gold)
        str_hyp (str): str de la phrase hypothèse (prediction)
        model (_type_, optional): _description_. Defaults to None. Not Used
        tokenizer (_type_, optional): _description_. Defaults to None. Not Used

    Returns:
        EditDistance: Objet EditDistance contenant le nombre d'erreur d'insertion, de suppression et de substitution
                        ainsi que la taille du tenseur de référence et la liste des alignements
    """
    global __ed_dict__
    if len(__ed_dict__) > 100000:   # NOTE: avoid the dictionary size to increase too much
        __ed_dict__ = {}

    rtoks = str_ref.split()
    htoks = str_hyp.split()
    for t in rtoks + htoks:
        if t not in __ed_dict__:
            __ed_dict__[t] = len(__ed_dict__)
    ref = torch.LongTensor( [__ed_dict__[t] for t in rtoks] )
    hyp = torch.LongTensor( [__ed_dict__[t] for t in htoks] )

    # These are 3, 3 and 4 in sclite if I remember well. However it does not change much the final error rate
    ins_weight = 1.0
    del_weight = 1.0
    sub_weight = 1.0

    curr_x_size = ref.size(0)
    curr_y_size = hyp.size(0)
    tsr_ed_matrix = torch.FloatTensor(curr_x_size+1, curr_y_size+1).fill_(0)
    for i in range(1, curr_x_size+1):
        tsr_ed_matrix[i,0] = float(i)
    for i in range(1, curr_y_size+1):
        tsr_ed_matrix[0,i] = float(i)

    for ij in range(curr_x_size*curr_y_size):
        i = (ij // curr_y_size)+1
        j = (ij % curr_y_size)+1
                            
        tmp_weight = 0
        if ref[i-1] != hyp[j-1]:
            tmp_weight = sub_weight 
        tsr_ed_matrix[i,j] = min(tsr_ed_matrix[i-1,j] + del_weight, tsr_ed_matrix[i,j-1] + ins_weight, tsr_ed_matrix[i-1,j-1] + tmp_weight)

    # Back-tracking for error rate computation
    n_ins = 0
    n_del = 0
    n_sub = 0

    alignement = []
    back_track_i = curr_x_size
    back_track_j = curr_y_size
    while back_track_i > 0 and back_track_j > 0:
        
        i = back_track_i
        j = back_track_j
        tmp_weight = 0
        if tsr_ed_matrix[i-1,j-1] != tsr_ed_matrix[i,j]:
            tmp_weight = sub_weight

        if tsr_ed_matrix[i-1,j] < tsr_ed_matrix[i,j-1]:
            if tsr_ed_matrix[i-1,j] < tsr_ed_matrix[i-1,j-1]:
                alignement.append( WordAlignement(alignement_type='del', 
                                                  index_reference=back_track_i-1, 
                                                  index_hypothese=None) )
                n_del += 1
                back_track_i -= 1
            else:
                back_track_i -= 1
                back_track_j -= 1
                alignement.append( WordAlignement(alignement_type='match', 
                                                  index_reference=back_track_i, 
                                                  index_hypothese=back_track_j) )
                if tmp_weight > 0:
                    n_sub += 1
                    alignement[-1] = WordAlignement(alignement_type='sub', 
                                                    index_reference=alignement[-1].index_reference, 
                                                    index_hypothese=alignement[-1].index_hypothese)

        else:   # tsr_ed_matrix[i-1,j] >= tsr_ed_matrix[i,j-1]
            if tsr_ed_matrix[i,j-1] < tsr_ed_matrix[i-1,j-1]:
                alignement.append( WordAlignement(alignement_type='ins', 
                                                  index_reference=back_track_i-1, 
                                                  index_hypothese=back_track_j-1) )
                n_ins += 1
                back_track_j -= 1
            else:
                back_track_i -= 1
                back_track_j -= 1
                alignement.append( WordAlignement(alignement_type='match', 
                                                  index_reference=back_track_i, 
                                                  index_hypothese=back_track_j) )
                if tmp_weight > 0:
                    n_sub += 1
                    alignement[-1] = WordAlignement(alignement_type='sub',
                                                    index_reference=alignement[-1].index_reference, 
                                                    index_hypothese=alignement[-1].index_hypothese)

    #print('[DEBUG] i and j before last phase: {}, {}'.format(back_track_i, back_track_j))

    while back_track_i > 0:
        #print('[DEBUG] adding del, {}, -'.format(back_track_i-1))

        alignement.append( WordAlignement(alignement_type='del', 
                                          index_reference=back_track_i-1, 
                                          index_hypothese=None) )
        back_track_i -= 1
        n_del += 1

    while back_track_j > 0:
        #print('[DEBUG] adding ins, {}, {}'.format(back_track_i-1, back_track_j-1))

        alignement.append( WordAlignement(alignement_type='ins', 
                                          index_reference=back_track_i, 
                                          index_hypothese=back_track_j-1) )
        back_track_j -= 1
        n_ins += 1

    #print('[DEBUG] alignement before reverse: {}'.format(alignement))

    alignement.reverse()
    return EditDistance(nombre_erreur_insertion=n_ins,
                         nombre_erreur_suppression= n_del,
                         nombre_erreur_substitution=n_sub, 
                         taille_tenseur_reference=curr_x_size, 
                         alignements=alignement)
    

def main(args):

    ref_str = args[1]
    hyp_str = args[2]

    print(' * Computing edit distance between:')
    print(' * ref: {}'.format(ref_str))
    print(' * hyp: {}'.format(hyp_str))
    print(' ---')

    er_vals = str_edit_distance(ref_str, hyp_str)

    print(' * ER: {:.2f}'.format(er_vals.get_wer()))# sum(er_vals[:3])/er_vals[3]))
    print(' * Errors:')
    print(' * ins: {}'.format(er_vals.nombre_erreur_insertion))
    print(' * del: {}'.format(er_vals.nombre_erreur_suppression))
    print(' * sub: {}'.format(er_vals.nombre_erreur_substitution))
    print(' ---')

    rtoks = ref_str.split()
    htoks = hyp_str.split()
    alignement = er_vals.alignements
    print( '* WordAlignement:' )
    for t in alignement:
        print(' * {}) r:{}, h:{}'.format(t[0], rtoks[t[1]], htoks[t[2]] if t[2] is not None else '-'))
    print(' ---')


if __name__ == '__main__':
    main(sys.argv)
