import os
import sys
import re
import json

import torch
import argparse
import xml.etree.cElementTree as ET

import edit_distance
from Classes.EditDistance import EditDistance

# Activate the following python environment for importing the German BERT:
# source /home/getalp/dinarelm/anaconda3/bin/activate ssl_wav2vec2_torch18

#from transformers import AutoModel, AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')
#model = AutoModel.from_pretrained('dbmdz/bert-base-german-cased')

_VERBOSE = True
_CORPUS_SYSTEM_COREF_MATCHES = False
_CORPUS_SYSTEM_COMPARISON_LOG = False
_MENTION_LOG = False
_DEBUG_LOG = True
_FULL_MATRICE = True
_DEBUG_WER = False
_CTX_NEEDED_AND_HARD_COREF_CONCAT_IDS = False # Permet de se restreindre à un subset de test
_PUDB=False
if _PUDB:
    import pudb; pudb.set_trace()
parser = argparse.ArgumentParser(description='Performs alignment between corpus (ParCorFul2) data and system data and compute coreference resolution metrics over coreference links using attention weights as scores')
parser.add_argument('corpus_source', help='source language corpus data')
parser.add_argument('corpus_target', help='target language corpus data')
parser.add_argument('system_data', help='system data, either input or output (specified by --evaluate-language), to align to corpus data')
parser.add_argument('--evaluate-language', type=str, default='source', help='Specify which language is evaluated: source (default), target')
parser.add_argument('--canmt-system', type=str, default='concat', help='Specify which type of CA-NMT is evaluated: concat (default), multienc')
parser.add_argument('--output-file', type=str, default="./attention_analysis.results", help='Specify the path of the output file')
parser.add_argument('--pudb', type=str, default="False", help='Specify the use of pudb for debugging. Default: False')
parser.add_argument('--local', type=str, default="False", help='Specifyif the script is run on the serveurs or not. Default: False')

args = parser.parse_args()


wer_threshold = 0.5
coreference_link_score = 'max'  # 'max' or 'avg', but the script only apply use_avg_score = coreference_link_score == 'avg'
canmt_system = args.canmt_system   # 'multienc' or 'concat', but the script only apply cur_bogus_idx = 0 if canmt_system == 'concat' else 1
eval_language = args.evaluate_language    # 'source' or 'target'
output_file = args.output_file # output file
if args.pudb == "True":
    import pudb; pudb.set_trace()
if eval_language == 'target':
    wer_threshold = 1000.0

_LOCAL=False if args.local == "False" else True
_PATH= "/home/getalp/lopezfab/lig/Attention_git/" if _LOCAL else "/home/getalp/lopezfab/Attention_git/"
_DATAPATH=f"{_PATH}/marco_script/data"


def read_txt(filename):

    f = open(filename, encoding='utf-8')
    ll = f.readlines()
    f.close()
    return [l.strip() for l in ll]

def pref2raw(prefix):

    tt = prefix.split('_')
    assert len(tt) == 2
    raw = '00' + tt[1]
    if len(raw) == 5:
        raw = '0' + raw
    return raw

def en_tokenize( text ):
    """
    Ad-hoc tokenization to have raw and tokenized sentences in the corpus match
    """

    punctuation = ['.', ',', ';', ':', '!', '?', '"', '\'', '[', ']']
    res = text.strip()
    tok_str = ' '.join(re.split(r'(\.\.\.|\.|,|;|:|\!|\?|"|\'|\[|\]|„|“|‚|‘|”|«|»|\(|\))', res))

    tok_str = tok_str.replace(' \' s ', ' \'s ')
    tok_str = tok_str.replace(' \' d ', ' \'d ')
    tok_str = tok_str.replace(' \' m ', ' \'m ')
    tok_str = tok_str.replace('n \' t ', ' n\'t ')
    tok_str = tok_str.replace(' \' re ', ' \'re ')
    tok_str = tok_str.replace(' \' ve ', ' \'ve ')
    tok_str = tok_str.replace(' \' ll ', ' \'ll ')
    tok_str = tok_str.replace(' \' 60s ', ' \'60s ')
    tok_str = tok_str.replace(' \' 70s ', ' \'70s ')
    tok_str = tok_str.replace(' \' 80s ', ' \'80s ')
    tok_str = tok_str.replace(' \' 90s ', ' \'90s ')
    tok_str = tok_str.replace(' D . S ', ' D.S ')
    tok_str = tok_str.replace(' U . S . ', ' U.S. ')
    tok_str = tok_str.replace(' D . C . ', ' D.C. ')
    tok_str = tok_str.replace(' o \' clock', ' o\'clock')
    tok_str = tok_str.replace(' Dr . ', ' Dr. ')
    tok_str = tok_str.replace(' mid- \'90s ', ' mid-\'90s ')
    tok_str = tok_str.replace(' I . P ', ' I.P ')
    tok_str = tok_str.replace(' P . C ', ' P.C ')
    tok_str = re.sub(' (\d+) , (\d\d\d)', ' \\1,\\2', tok_str )
    tok_str = re.sub('(\d+) . (\d+)', '\\1.\\2', tok_str)
    tok_str = tok_str.replace(' Amazon . com', ' Amazon.com')
    tok_str = tok_str.replace(' cannot', ' can not')
    tok_str = tok_str.replace(' U . K .', ' U.K.')
    tok_str = tok_str.replace(' Ph . D . ', ' Ph.D. ')
    tok_str = re.sub(' \' (\d\d) ', ' \'\\1 ', tok_str)
    tok_str = re.sub(' \$(\d+) ', ' $ \\1 ', tok_str)
    tok_str = tok_str.replace(' wanna', ' wan na')
    tok_str = tok_str.replace(' E . T . ', ' E.T. ')
    # Ad hoc processing for a specific sentence:
    if 'unbelievable movie about what' in tok_str:
        tt = tok_str.split()
        new_str = []
        first = True
        for t in tt:
            if t == 'E.T.' and first:
                new_str.append(t)
                first = False
            elif t == 'E.T.':
                new_str.append('E.T')
                new_str.append('.')
            else:
                new_str.append(t)
        tok_str = ' '.join(new_str) 
    if 'divorce-crippled family' in tok_str:
        tok_str = tok_str.replace('E.T.', 'E.T .')
    tok_str = tok_str.replace(' L . A . ', ' L.A. ')
    tok_str = tok_str.replace(' C \' mere ', ' C\'mere ')
    tok_str = tok_str.replace(' \' Cause ', ' \'Cause ')
    tok_str = tok_str.replace(' C \' mon ', ' C\'mon ')
    tok_str = tok_str.replace(' gotta ', ' got ta ')
    tok_str = tok_str.replace(' Oh-- ', ' Oh -- ')
    tok_str = tok_str.replace(' \' cause ', ' \'cause ')
    tok_str = tok_str.replace( ' gonna ', ' gon na ')

    if tok_str[-5:] == 'U.K. ':
        tok_str = tok_str[:-5] + 'U.K .'
    if tok_str[-5:] == 'U.S. ':
        tok_str = tok_str[:-5] + 'U.S .'
    tok_str = tok_str.replace('  ', ' ')
    tok_str = tok_str.replace(' \' alliance of misfits', ' \'alliance of misfits')
    tok_str = tok_str.replace(' St . Petersburg ', ' St. Petersburg ' )
    tok_str = re.sub( '(\d+)\%', '\\1 %', tok_str )
    tok_str = tok_str.replace('can be \' photographed', 'can be \'photographed')
    tok_str = tok_str.replace('or \' captured', 'or \'captured')
    tok_str = tok_str.replace('www . drjoetoday . com', 'www.drjoetoday.com')
    tok_str = tok_str.replace('Stefanie R . Ellis', 'Stefanie R. Ellis')
    tok_str = re.sub('\$(\d+).(\d+)', '$ \\1,\\2', tok_str)
    tok_str = tok_str.replace('a . m .', 'a.m.')
    tok_str = tok_str.replace('p . m .', 'p.m.')
    tok_str = re.sub(' #(\d+)', ' # \\1', tok_str)

    return tok_str.strip()

def de_tokenize( text ):
    """
    Ad-hoc tokenization to have raw and tokenized sentences in the corpus match
    """

    tok_str = en_tokenize( text )
    #tok_str = re.sub(' (\d)+\.(\d+) ', ' \\1,\\2 ', tok_str)
    tok_str = tok_str.replace(' Die sind 1.80 Meter ', ' Die sind 1,80 Meter ')
    tok_str = tok_str.replace(' Wirtschaftswachstum um 1.3 Prozent ', ' Wirtschaftswachstum um 1,3 Prozent ')
    tok_str = tok_str.replace(' du hättest 1.7 Millionen Dokumente ', ' du hättest 1,7 Millionen Dokumente ')
    tok_str = tok_str.replace(' Mindestlohn von 7.25 US-Dollar sei ', ' Mindestlohn von 7,25 US-Dollar sei ')
    #tok_str = tok_str.replace(' mindestens 50 . ten ', ' mindestens 50.ten ')
    tok_str = tok_str.replace(' U.S. -Außenministeriums ', ' U.S.-Außenministeriums ' )
    #tok_str = tok_str.replace(' 1,000 US-Dollar ', ' 1.000 US-Dollar ')
    if tok_str == 'Z . B . in der Telekommunikation können Sie die gleiche Geschichte über Glasfaser erklären .':
        tok_str = 'Z. B. in der Telekommunikation können Sie die gleiche Geschichte über Glasfaser erklären .'
    if tok_str == 'Es bedeutet z . B . , dass wir ausarbeiten müssen ,  wie man Zusammenarbeit und Konkurrenz gleichzeitig unterbringt .':
        tok_str = 'Es bedeutet z. B. , dass wir ausarbeiten müssen ,  wie man Zusammenarbeit und Konkurrenz gleichzeitig unterbringt .'
    #tok_str = tok_str.replace(' 4,000 Überwachungsanträge ', ' 34.000 Überwachungsanträge ')
    tok_str = tok_str.replace('Amazon . com ', 'Amazon.com ')
    tok_str = tok_str.replace(' Mio . ', ' Mio. ')
    tok_str = tok_str.replace(' etc .', ' etc.')
    tok_str = tok_str.replace('Google X .', 'Google X.')
    tok_str = tok_str.replace('N . Negroponte', 'N. Negroponte')
    tok_str = tok_str.replace(' usw .', ' usw.')
    tok_str = tok_str.replace('ABC " -Lieder', 'ABC"-Lieder')
    tok_str = tok_str.replace('wie z . B . die Druckmaschine', 'wie z.B. die Druckmaschine')
    tok_str = tok_str.replace('Und wenn man sich z . B .  ', 'Und wenn man sich z.B. ')
    tok_str = tok_str.replace(' ist . ¾', ' ist.¾')
    tok_str = tok_str.replace('ist heute überall . .', 'ist heute überall ..')
    if tok_str == 'So .':
        tok_str = 'So.'
    tok_str = tok_str.replace('wie z . B . Ibrahim Böhme', 'wie z. B. Ibrahim Böhme')
    #tok_str = tok_str.replace('Preisgeld von 5,000 Euro gleich weiter', 'Preisgeld von 25.000 Euro gleich weiter')
    tok_str = tok_str.replace('St . Petersburger', 'St. Petersburger')
    #tok_str = tok_str.replace('3 . 000ten', '3.000ten')
    tok_str = re.sub(' (\d+) . (\d+)?ten ', ' \\1.\\2ten ', tok_str)
    tok_str = tok_str.replace('mit 0,000 Leuten', 'mit 20.000 Leuten')
    #tok_str = tok_str.replace(' 4,300 ', ' 4.300 ')
    tok_str = tok_str.replace('zu Putin : " Danke ', 'zu Putin:"Danke ')
    tok_str = tok_str.replace(' 1.35 Mrd . ', ' 1,35 Mrd. ')
    #tok_str = tok_str.replace(' ca . 6.000 Tonnen ', ' ca. 6.000 Tonnen ')
    tok_str = re.sub(' ca . (\d)+', ' ca. \\1', tok_str)
    tok_str = tok_str.replace( 'www . drjoetoday . com', 'www.drjoetoday.com' )
    tok_str = tok_str.replace(' Final Five " -Mannschaftskameradin ', ' Final Five"-Mannschaftskameradin ')
    tok_str = tok_str.replace( 'Stefanie R . Ellis', 'Stefanie R. Ellis' )
    tok_str = tok_str.replace( 'von 13.75 Zoll Regen', 'von 13,75 Zoll Regen' )
    tok_str = re.sub( ' #(\d+) ', ' # \\1 ', tok_str )
    tok_str = re.sub( ' £(\d+)', ' £ \\1', tok_str)
    tok_str = re.sub( ' (\d+)mg ', ' \\1 mg ', tok_str )
    tok_str = tok_str.replace('Aber in St. Petersburg lautete', 'Aber in St . Petersburg lautete')

    return tok_str

def get_words(filename, key_prefix):

    words = {}
    for word in ET.parse(filename).getroot():
        key = key_prefix + '-' + word.attrib['id']
        words[key] = word.text
    return words

def get_span_idx( ss ):

    idxs = []
    tt = ss.replace('..', ' ').split()
    for t in tt: 
        if 'word_' in t:
            idxs.append(int(t[5:])) 
    if len(idxs) > 1:
        assert len(idxs) == 2
        idxs = list(range(idxs[0],idxs[1]+1))
    return idxs

def get_all_spans( ss ):

    idxs = []
    spans = ss.replace(',', ' ').split()
    for span in spans:
        ii = get_span_idx( span )
        idxs.append( ii )
    return idxs

def get_mention_from_idxs( prefix, idxs, words ):
    return ' '.join([words[prefix + '-word_'+str(ii)] for ii in idxs])

def read_discomt_data():
    """
    Returns 2 dicionary structures, one for src data and one for tgt data.
    The dictionaries contain keys:
        'text' (map from raw text to tokenized text, see below);
        'words' (map from uniq word ID to word in the tokenized text, see below);
        'coref' (list with dictionary elements with all the information for each coreference mention, see below)
        'words_in_coref' (map from uniq word ID to entity ID to indentify words that actually appears in coreferent mentions, see below).
    """

    datapath = f'{_DATAPATH}/ParCorFull2/parcor-full/corpus/DiscoMT/'
    src_data_path = datapath + 'EN/'
    tgt_data_path = datapath + 'DE/'
    prefixes = ['000_1756', '001_1819', '002_1825', '003_1894', '005_1938', '006_1950', '007_1953', '009_2043', '010_205', '011_2053']
    raw_txt_path = 'Source/sentence/'
    word_path = 'Basedata/'
    markable_path = 'Markables/' 

    # src_wer = [0, 0, 0, 0]
    src_wer = EditDistance(nombre_erreur_insertion=0, nombre_erreur_suppression=0, nombre_erreur_substitution=0, taille_tenseur_reference=0, alignements=[])
    # tgt_wer = [0, 0, 0, 0]
    tgt_wer = EditDistance(nombre_erreur_insertion=0, nombre_erreur_suppression=0, nombre_erreur_substitution=0, taille_tenseur_reference=0, alignements=[])

    src_data = {}
    tgt_data = {}
    src_raw2tok_text = {}   # NOTE: mapping from raw sentences in the corpus to tokenized sentences in the corpus,
                            # together with the alignment between the 2.
    tgt_raw2tok_text = {}   # NOTE: same for the target-side
    # 1. Read src and tgt raw texts
    for pp in prefixes:
        rp = pref2raw(pp)
        filename = src_data_path + raw_txt_path + 'talk' + rp + '.de-en.en'
        f = open(filename, encoding='utf-8')
        src_ll = f.readlines()
        f.close()
        filename = tgt_data_path + raw_txt_path + 'talk' + rp + '.de-en.de'
        f = open(filename, encoding='utf-8')
        tgt_ll = f.readlines()
        f.close()
        assert len(src_ll) == len(tgt_ll)
        for s, t in zip(src_ll, tgt_ll):    # [(src_ll[i], tgt_ll[i] for i in range(len(src_ll)))]
            if len(s.strip()) > 0:
                assert len(t.strip()) > 0

                # 1. Source-side raw-to-tokenized sentence alignment
                tok_s = en_tokenize(s)
                ref = tok_s
                hyp = s
                er_vals = edit_distance.str_edit_distance(ref, hyp)
                src_wer += er_vals
                a  = er_vals.alignements
                assert len(a) == len(ref.split())
                src_raw2tok_text[s.strip()] = (tok_s, a)

                # 2. Target-side raw-to-tokenized sentence alignment
                tok_t = de_tokenize(t)
                ref = tok_t
                hyp = t
                er_vals = edit_distance.str_edit_distance(ref, hyp)
                tgt_wer += er_vals
                a = er_vals.alignements
                assert len(a) == len(ref.split())
                tgt_raw2tok_text[t.strip()] = (tok_t, a)

    src_data['text'] = src_raw2tok_text
    tgt_data['text'] = tgt_raw2tok_text

    if _DEBUG_WER:
        print('[DEBUG-WER] disco-mt corpus source-side raw-to-tokenized WER: {:.2f}'.format(src_wer.get_wer() *100))
        print('[DEBUG-WER] disco-mt corpus target-side raw-to-tokenized WER: {:.2f}'.format(tgt_wer.get_wer() *100))
        sys.stdout.flush()

    # 2. Read src and tgt word files
    src_words = {}  # NOTE: mapping from a key constituted by <file ID>-<word ID>, where file ID is a unique file ID, word ID is the word ID in the file (as specified in XML format).
    tgt_words = {}  # NOTE: same for target-size words
    for pp in prefixes:
        filename = src_data_path + word_path + pp + '_words.xml'
        src_words.update( get_words(filename, pp) )
        filename = tgt_data_path + word_path + pp + '_words.xml'
        tgt_words.update( get_words(filename, pp) )
    src_data['words'] = src_words
    tgt_data['words'] = tgt_words

    # 3. Read src and tgt coreferences
    src_coref = []  # NOTE: each element is a dictionary where keys are the same as in the XML file containing coreference annotations, in particular the span indices and the entity ID
    tgt_coref = []  # NOTE: same for the target side
    src_words_in_coref = {} # NOTE: mapping from uniq word ID to entity ID
    tgt_words_in_coref = {} # NOTE: same for target words
    for pp in prefixes:
        filename = src_data_path + markable_path + pp + '_coref_level.xml'
        ff_coref, words_in_coref = get_coreferences(filename, pp)
        src_coref.extend( ff_coref )
        src_words_in_coref.update( words_in_coref )
        filename = tgt_data_path + markable_path + pp + '_coref_level.xml'
        ff_coref, words_in_coref = get_coreferences(filename, pp)
        tgt_coref.extend( ff_coref )
        tgt_words_in_coref.update( words_in_coref )
    src_data['coref'] = src_coref
    tgt_data['coref'] = tgt_coref
    src_data['words_in_coref'] = src_words_in_coref
    tgt_data['words_in_coref'] = tgt_words_in_coref

    ###############################################
    return src_data, tgt_data
    ###############################################

    # debug test
    for k in src_coref_by_set.keys():
        pp = k.split('-')[0]
        print(' ###COREF set {}'.format(k))
        for cc in src_coref_by_set[k]:
            span = cc['span']
            idxs = get_all_spans(span)

            mentions = []
            for ii in idxs:
                ii.sort()
                for idx in ii:
                    word = src_words[pp + '-word_'+str(idx)] 
                mentions.append( get_mention_from_idxs(pp, ii, src_words) )
            print(' * ###COREF mention {}'.format( '|'.join(mentions)))
            print(' * ###COREF ---')
        print(' ###COREF **********')

    return src_data, tgt_data

def read_news_data():
    """
    Same as read_discomt_data function, but for the news data part of the ParCorFull 2 corpus.
    Why 2 functions doing the same thing ? I don't know... file ID format is different, but still ...
    """

    datapath = f'{_DATAPATH}/ParCorFull2/parcor-full/corpus/news/'
    src_datapath = datapath + 'EN/'
    tgt_datapath = datapath + 'DE/'
    prefixes = ['03', '04', '05', '07', '08', '09', '10', '13', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25']
    raw_txt_path = 'Source/'
    word_path = 'Basedata/'
    markable_path = 'Markables/'

    src_wer = EditDistance(nombre_erreur_insertion=0, nombre_erreur_suppression=0, nombre_erreur_substitution=0, taille_tenseur_reference=0, alignements=[])
    tgt_wer = EditDistance(nombre_erreur_insertion=0, nombre_erreur_suppression=0, nombre_erreur_substitution=0, taille_tenseur_reference=0, alignements=[])

    src_data = {}
    tgt_data = {}

    # 1. Read raw src and tgt texts
    src_raw2tok_data = {}
    tgt_raw2tok_data = {}
    src_ll = []
    tgt_ll = []
    for pp in prefixes:
        filename = src_datapath + raw_txt_path + pp + '.en.xml'
        for paragraph in ET.parse(filename).getroot(): 
            for seg in paragraph:
                src_ll.append( seg.text.strip() )
        filename = tgt_datapath + raw_txt_path + pp + '.de.xml'
        for paragraph in ET.parse(filename).getroot():
            for seg in paragraph:
                tgt_ll.append( seg.text.strip() )
    assert len(src_ll) == len(tgt_ll)
    for s, t in zip(src_ll, tgt_ll):
        tok_s = en_tokenize(s)
        ref = tok_s
        hyp = s
        er_vals = edit_distance.str_edit_distance(ref, hyp)
        src_wer += er_vals
        a = er_vals.alignements
        src_raw2tok_data[s] = (tok_s, a)
        tok_t = de_tokenize(t)
        ref = tok_t
        hyp = t
        er_vals = edit_distance.str_edit_distance(ref, hyp)
        tgt_wer += er_vals
        a = er_vals.alignements
        tgt_raw2tok_data[t] = (tok_t, a)
    src_data['text'] = src_raw2tok_data
    tgt_data['text'] = tgt_raw2tok_data

    if _DEBUG_WER:
        print('[DEBUG-WER] news corpus source-side raw-to-tokenized WER: {:.2f}'.format(src_wer.get_wer() *100))
        print('[DEBUG-WER] news corpus target-side raw-to-tokenized WER: {:.2f}'.format(tgt_wer.get_wer() *100))
        sys.stdout.flush()

    # 2. Read src and tgt words
    src_words = {}
    tgt_words = {}
    for pp in prefixes:
        filename = src_datapath + word_path + pp + '_words.xml'
        src_words.update( get_words(filename, pp) )
        filename = tgt_datapath + word_path + pp + '_words.xml'
        tgt_words.update( get_words(filename, pp) )
    src_data['words'] = src_words
    tgt_data['words'] = tgt_words

    # 3. Read src and tgt coreferences
    src_coref = []
    tgt_coref = []
    src_words_in_coref = {}
    tgt_words_in_coref = {}
    for pp in prefixes:
        filename = src_datapath + markable_path + pp + '_coref_level.xml'
        ff_coref, words_in_coref = get_coreferences(filename, pp)
        src_coref.extend( ff_coref )
        src_words_in_coref.update( words_in_coref )
        filename = tgt_datapath + markable_path + pp + '_coref_level.xml'
        ff_coref, words_in_coref = get_coreferences(filename, pp)
        tgt_coref.extend( ff_coref )
        tgt_words_in_coref.update( words_in_coref )

    src_data['coref'] = src_coref
    tgt_data['coref'] = tgt_coref
    src_data['words_in_coref'] = src_words_in_coref
    tgt_data['words_in_coref'] = tgt_words_in_coref

    return src_data, tgt_data

def get_coreferences(filename, key_prefix):

    if _DEBUG_LOG:
        print('[DEBUG]get_coreferences, extracting mentions from file {}'.format(filename))
        sys.stdout.flush()

    corefs = []
    words_in_corefs = {}
    for word in ET.parse(filename).getroot():
        #print('[DEBUG]get_coreferences, got markable: id {}; span: {}; coref_class: {}'.format(word.attrib['id'], word.attrib['span'], word.attrib['coref_class']))
        #sys.stdout.flush()

        corefs.append( {'id': word.attrib['id'], 'fileid': key_prefix, 'span': word.attrib['span'], 'coref_class': word.attrib['coref_class'], 'mention': word.attrib['mention']} )
        idxs = get_all_spans( word.attrib['span'] )
        for span in idxs:
            for idx in span: 
                key = key_prefix + '-word_' + str(idx)
                #assert key not in words_in_corefs, 'key {} is already defined (keys should be uniques at file level)'.format(key)
                words_in_corefs[key] = word.attrib['coref_class']

                if _DEBUG_LOG:
                    print('[DEBUG]get_coreferences: adding words_in_corefs with key {}'.format(key))
                    sys.stdout.flush()
    return corefs, words_in_corefs

def match_nmt2parcorfull(src_nmt, disco_src_pcf, news_src_pcf):
    """
    Matches reference sentences used for the NMT system (that is not, for the target side, the sentences generated by the system) 
    with sentences in one of the two ParCorFull 2 corpora (disco and news)
    
    Returns :
        List[List]: a list where each element is a list itself containing:
            0. The NMT system input/output sentence : type = str (exemple: '"Even in purely non-religious terms, homosexuality represents a misuse of the sexual faculty.')
            1. The corresponding raw ParCorFull 2 corpus sentence : type = str (exemple: '" Even in purely non-religious terms , homosexuality represents a misuse of the sexual faculty .')
            2. The sequence of uniq token identifiers for tokens in the corpus sentence : type = List[str] (exemple: ['000_1756-word_1', '000_1756-word_2', '000_1756-word_3', '000_1756-word_4', '000_1756-word_5', '000_1756-word_6', '000_1756-word_7', '000_1756-word_8', '000_1756-word_9', '000_1756-word_10', '000_1756-word_11', '000_1756-word_12', '000_1756-word_13', '000_1756-word_14', '000_1756-word_15', '000_1756-word_16'])
            3. The corresponding tokenized ParCorFull 2 corpus sentence : type = List[str] (exemple: [('del', 0, None), ('sub', 1, 0), ('match', 2, 1), ('match', 3, 2), ('match', 4, 3), ('del', 5, None), ('sub', 6, 4), ('match', 7, 5), ('match', 8, 6), ('match', 9, 7), ('match', 10, 8), ('match', 11, 9), ('match', 12, 10), ('match', 13, 11), ('del', 14, None), ('sub', 15, 12)])
            4. The same sentence as in 1. annotated with mentions, in the form of tokens "decorated" with [<token>]set_idxxx : type = str (exemple: '#[``]#-set_219 #[Even]#-set_219 #[in]#-set_219 #[purely]#-set_219 #[non-religious]#-set_219 #[terms]#-set_219 #[,]#-set_219 #[homosexuality]#-set_219 #[represents]#-set_219 #[a]#-set_219 #[misuse]#-set_219 #[of]#-set_219 #[the]#-set_219 #[sexual]#-set_219 #[faculty]#-set_219 #[.]#-set_219')
    """

    matches = 0
    mismatches = 0
    matched_data = []
    found = {}
    disco_src_text = disco_src_pcf['text']
    news_src_text = news_src_pcf['text']
    for s in src_nmt:
        if s in disco_src_text:
            #print(' ### MATCH: {}'.format(s))
            #sys.stdout.flush()
            matches += 1
            if s not in found:
                found[s] = 1
            else:
                if _VERBOSE:
                    print(' ### CLASH (DiscoMT): {} already found!'.format(s))
        elif s in news_src_text:
            matches += 1
            if s not in found:
                found[s] = 1
            else:
                if _VERBOSE:
                    print(' ### CLASH (news): {} already found!'.format(s))
        else:
            mismatches += 1

    if _VERBOSE:
        print(' ### Found {} out of {} matched sentences'.format(matches, len(src_nmt)))
        sys.stdout.flush()

    def get_coref_annotated_sentence(wids, data):

        # For DEBUGGING
        coref_words = 0

        res = []
        for wid in wids:
            if wid in data['words_in_coref']:
                res.append( '#[' + data['words'][wid] + ']#-' + data['words_in_coref'][wid] )

                # For DEBUGGING
                coref_words += 1
            else:
                res.append( data['words'][wid] )

        if _DEBUG_LOG:
            print('[DEBUG]get_coref_annotated_sentence, words in coreferent mentions over total words: {} / {}'.format(coref_words, len(wids)))
            sys.stdout.flush()

        return ' '.join(res)

    disco_word_idx = 0
    news_word_idx = 0
    disco_words_pcf = list(disco_src_pcf['words'].items())
    news_words_pcf = list(news_src_pcf['words'].items())
    stop = False
    stop_word = ''
    for s in src_nmt:
        matched_struct = [s]    # NOTE: contains: 1. NMT sentence; 2. the corresponding corpus tokenized sentence; 3. the sequence of uniq word IDs; 4. the alignment between 1. and 2.; 5. the sentence in 2. annotated with coreferent mentions.
        if s in disco_src_text:
            tok_s = disco_src_text[s][0]
            wids = []
            for t in tok_s.split():
                w = disco_words_pcf[disco_word_idx][1]
                wids.append( disco_words_pcf[disco_word_idx][0] )
                if w != t and not (w == '``' and t == '"') and not (w == '\'\'' and t == '"'):
                    if _VERBOSE: 
                        print(' ### MISSMATCH@ {} vs. {}; previous {}; next {}'.format(w, t, disco_words_pcf[disco_word_idx-1][1], disco_words_pcf[disco_word_idx+1][1]))
                    stop = True
                    stop_word = w
                    break
                disco_word_idx += 1 
            #disco_src_text[s] = (disco_src_text[s][0], wids, disco_src_text[s][1], get_coref_annotated_sentence(wids, disco_src_pcf))

            if _DEBUG_LOG:
                print('[DEBUG]match_nmt2parcorfull(disco), getting annotated sentence with wids: {}'.format(wids))
                sys.stdout.flush()

            matched_struct.extend( [disco_src_text[s][0], wids, disco_src_text[s][1], get_coref_annotated_sentence(wids, disco_src_pcf)] )
            assert len(matched_struct[1].split()) == len(matched_struct[-1].split())
        elif s in news_src_text:
            tok_s = news_src_text[s][0]
            wids = []
            for t in tok_s.split():
                w = news_words_pcf[news_word_idx][1]
                wids.append( news_words_pcf[news_word_idx][0] )
                if w != t and not (w == '``' and t == '"') and not (w == '\'\'' and t == '"'):
                    if _VERBOSE:
                        print(' ### MISSMATCH {} vs. {}; previous {}; newt {}'.format(w, t, news_words_pcf[news_word_idx-1][1], news_words_pcf[news_word_idx+1][1]))
                    stop = True
                    stop_word = w
                    break
                news_word_idx += 1
            #news_src_text[s] = (news_src_text[s][0], wids, news_src_text[s][1], get_coref_annotated_sentence(wids, news_src_pcf))

            if _DEBUG_LOG:
                print('[DEBUG]match_nmt2parcorfull(news), getting annotated sentence with wids: {}'.format(wids))
                sys.stdout.flush()

            matched_struct.extend( [news_src_text[s][0], wids, news_src_text[s][1], get_coref_annotated_sentence(wids, news_src_pcf)] )
            assert len(matched_struct[1].split()) == len(matched_struct[-1].split())
        else:
            print(' ### sentence {} not defined!'.format(s))
            return None

        if stop:
            print(' ### word {} missmatch in {}'.format(stop_word, tok_s))
            return None
        else:
            if _VERBOSE:
                print(' * MATCH: {}'.format(tok_s))
                sys.stdout.flush()
        matched_data.append( matched_struct )
    print("[DEBUG] matched_data: {}".format(matched_data))
    return matched_data

def read_system_src_data(src_list, seq_ids=None):
    """
    Reads in system data.
    Returns a dictionary where keys are the sentence IDs (e.g. 102-1, 102-2, etc.) and values are themselves dictionaries containing:
     1. key 'cur': current input/output sentence
     2. key 'ctx': context input/output sentence
     3. key 'att': matrix of attention weights between current and context sentence
    """

    f = open(src_list, encoding='utf-8')
    ll = f.readlines()
    f.close()

    dispersion_stats = [0, 0]
    system_data = {}
    register_tokens = ['.', ',', '!', '?', '<end>', '<eos>', '<pad>']
    for ff in ll:
        f = open(ff.strip(), encoding='utf8')
        tt = f.readlines()
        f.close()

        ctx_tt = tt[0].strip().split('\t')
        seq_id = ctx_tt[0]
        ctx_seq = ' '.join(ctx_tt[1:])
        cur_seq = ' '.join([tt[i].strip().split('\t')[0] for i in range(1, len(tt))])

        weights = []
        for ii in range(1, len(tt)):
            ww = tt[ii].replace('\n', '').split('\t')[1:]
            ww = [float(w) if w != '.' else 0.0 for w in ww]
            weights.append( ww )

        if seq_ids is None or seq_id in seq_ids:
            assert seq_id not in system_data
            system_data[seq_id] = {'cur': cur_seq, 'ctx': ctx_seq, 'att': weights}

            for ww in weights:
                for ctx_idx, ctx_tok in enumerate(ctx_seq.split()):
                    dispersion_stats[0] += int(ww[ctx_idx] > 0.0) if ctx_tok not in register_tokens else 0 
            dispersion_stats[1] += len([t for t in cur_seq.split() if t not in register_tokens])

    if _VERBOSE:
        print(' * Source system data read, dispersion factor: {:.4f}'.format(dispersion_stats[0] / dispersion_stats[1]))
        sys.stdout.flush()

    return system_data

def compute_head_offset(idx, heads):

    res = idx
    h_idx = 1
    while idx+1 >= heads[h_idx]:
        res += 1
        h_idx += 1
    if res+1 == heads[h_idx]-1:
        res += 1

    return res

def find_coref_matches(src_sentence, sys_sentence, alignements):
    """
    Using the alignment alignements, computed with edit-distance, find tokens in the system input/output sentence corresponding to mention tokens in the corresponding gold, corpus sentence.

    * src_sentence: contains the raw sentence in the ParCorFull 2 corpus, the tokenized sentence, the sequence of token IDs in the corpus, and the sentence annotated with mentions
    * sys_sentence: the system input/output sentence
    * alignements: the alignement computed with edit-distance between the tokenized gold sequence (src_sentence[1]) and the system input/output sequence

    Returns a list of triples (as tuples), where elements are respectively:
     - 1. the index of the aligned token in the system input/output sequence
     - 2. the set_id of the mention (the cluster)
     - 3. True if the aligned token is identical to the token in the gold sentence.
    """

    if _CORPUS_SYSTEM_COREF_MATCHES:
        print('[DEBUG]find_coref_matches src_s: {}'.format(src_sentence))
        print('[DEBUG]find_coref_matches sys_s: {}'.format(sys_sentence))
        sys.stdout.flush()

    sys_coref_idxs = []
    src_tokenized_sentence = src_sentence[1].split()
    src_annoted_sentence = src_sentence[-1].split()
    sys_tokenized_sentence = sys_sentence.split()
    for alignement in alignements:
        #print(' *** [DEBUG]-COREF, alignment entry: {}'.format(e))
        if alignement.index_reference is not None \
            and alignement.index_hypothese is not None \
            and src_annoted_sentence[alignement.index_reference][:2] == '#[' and ']#-set' in src_annoted_sentence[alignement.index_reference] \
            and sys_tokenized_sentence[alignement.index_hypothese] != '<pad>':
            set_id = src_annoted_sentence[alignement.index_reference].split('-')[-1]
            assert 'set_' in set_id
            sys_coref_idxs.append( (alignement.index_hypothese,
                                    set_id,
                                    src_tokenized_sentence[alignement.index_reference] == sys_tokenized_sentence[alignement.index_hypothese]
                                ) )

            if _CORPUS_SYSTEM_COREF_MATCHES:
                print(' * [DEBUG]find_coref_matches COREF-MATCH got ed op {} for mention tokens in entity {}: {} vs. {}'.format(
                        alignement.alignement_type, 
                        set_id, 
                        src_tokenized_sentence[alignement.index_reference] if alignement.index_reference is not None else '-', 
                        sys_tokenized_sentence[alignement.index_hypothese] if alignement.index_hypothese is not None else '-'
                    ),
                    flush=True
                )
    return sys_coref_idxs

def find_coref_links(sys_s, s_corefs, sys_c, c_corefs, att):
    """_summary_

    Args:
        sys_s (str): string of source current sentence from the system
        s_corefs (tuple): triples (as tuples), where elements are respectively:
                            - 1. the index of the aligned token in the system input/output sequence     
                            - 2. the set_id of the mention (the cluster)     
                            - 3. True if the aligned token is identical to the token in the gold sentence.
        sys_c (str): string of source context sentence from the system
        c_corefs (tuple): triples (as tuples), where elements are respectively:
                            - 1. the index of the aligned token in the system input/output sequence     
                            - 2. the set_id of the mention (the cluster)     
                            - 3. True if the aligned token is identical to the token in the gold sentence.
        att (List[List[float]]): List of List of attention weight between crt and ctx. Taille: len(crt) x len(ctx)

    Raises:
        ValueError: _description_

    Returns:
        tupple: tuple dont:
            0: system current sentence
            1: system context sentence
            2: metrics : List of éléments:
                0: (booléen) default to True
                1: link_score >= all_lines_max_weight
                2: sum(all_weights) > 0.0
                3: link_score
    """
    
    s_tt = sys_s.split()
    c_tt = sys_c.split()
    annot_info = []
    metrics = [False, False, 0.0]   
    # 1. Is max weight in the antecedent (any token) ?; 
    # 2. Is antecedent att weight > 0.0 (any token) ?; 
    # 3. Att weight to the antecedent

    def compute_link_score( weights, avg=False):
        if avg:
            return float(sum(weights))/float(len(weights))
        else:
            return max(weights)

    def get_mentions( corefs ):
        """
        corefs is the data struct constructed by the find_coref_matches function, it is a list containing triples, where each triple contains:
        - the index of the token in the system input/output sentence belonging to a mention
        - the set ID of the mention
        - True if the token is identical to the corresponding token in the reference corpus sentence as from alignment with edit-distance, False otherwise.
        
        Return:
        
        """

        if len(corefs) == 0:
            return {}

        start = corefs[0][0]
        cid = corefs[0][1]
        entities = {}
        curr_idxs = []

        if len(corefs) == 1:
            #assert corefs[0][2] # why I assert this ?
            entities[corefs[0][1]] = [[corefs[0][0]]]
        else:
            #if corefs[0][2]:    # NOTE: if the token is identical to the corresponding reference sentence token...
            curr_idxs.append( corefs[0][0] )

        for i in range(1, len(corefs)):
            if corefs[i-1][0] != corefs[i][0]-1 or corefs[i-1][1] != corefs[i][1]:  # NOTE: detect begin of a new mention: either token indices are not adjacent, or set IDs are different
                if corefs[i-1][1] not in entities:
                    entities[corefs[i-1][1]] = []
                if len(curr_idxs) > 0:
                    entities[corefs[i-1][1]].append( curr_idxs )
                start = corefs[i][0]
                curr_idxs = [start]
            if i == len(corefs)-1:  # NOTE: last token
                if corefs[i-1][0] != corefs[i][0]-1 or corefs[i-1][1] != corefs[i][1]:  # NOTE: same as above, detect begin of a new mention
                    assert start == corefs[i][0] and len(curr_idxs) == 1, 'start: {} vs. corefs[i][0]: {}; len(curr_idxs): {}'.format(start, corefs[i][0], len(curr_idxs))
                    if corefs[i][1] not in entities:
                        entities[corefs[i][1]] = []
                    #if corefs[i][2]:    # NOTE: if the token is identical to the corresponding reference sentence token...
                    curr_idxs = [corefs[i][0]]
                    #if len(curr_idxs) > 0:
                    entities[corefs[i][1]].append( curr_idxs )
                    curr_idxs = []  # Not needed!
                else:
                    if corefs[i][1] not in entities:
                        entities[corefs[i][1]] = []
                    #if corefs[i][2]:    # NOTE: if the token is identical to the corresponding reference sentence token...
                    curr_idxs.append( corefs[i][0] )
                    #if len(curr_idxs) > 0:
                    entities[corefs[i][1]].append( curr_idxs )

            #if corefs[i][2]:    # NOTE: if the token is identical to the corresponding reference sentence token...
            curr_idxs.append( corefs[i][0] )

        if _DEBUG_LOG:
            print('[DEBUG] get_mentions, found mentions (num. of mentions: {}): {}'.format(len(entities), entities))
            sys.stdout.flush()

        # Clean mentions from duplicate indices
        # TODO: also check there is no gap in consecutive indices !!!
        for set_id in entities:
            mentions = entities[set_id]
            for i in range(len(mentions)):
                mnt = mentions[i]
                dd = {}
                clean_mnt = []
                for idx in mnt:
                    if idx not in dd:
                        dd[idx] = 1
                        clean_mnt.append(idx)
                for idx in range(1, len(clean_mnt)):
                    if clean_mnt[idx]-1 != clean_mnt[idx-1]:
                        raise ValueError('[DEBUG-CHECK] found non adjacent indices in mention {}'.format(clean_mnt))

                mentions[i] = clean_mnt
            entities[set_id] = mentions

        return entities

    def get_mention_txt(seq_idxs, mention_idxs):
        return ' '.join( [seq_idxs[i] for i in mention_idxs] )

    if _DEBUG_LOG:
        print('[DEBUG] mentions in current sentences:')
        print(s_corefs)
        print('[DEBUG] **********')
        sys.stdout.flush()

    cur_mentions = get_mentions(s_corefs)
    if _MENTION_LOG:
        print(' * [DEBUG-MENTION] current sentence mentions:')
        for set_id in cur_mentions.keys():
            for mention in cur_mentions[set_id]:
                mention_txt = get_mention_txt(s_tt, mention)
                print(' * [DEBUG-MENTION] - Set {}: {}'.format(set_id, mention_txt))
                sys.stdout.flush()
        print(' * [DEBUG-MENTION] ----------')
        sys.stdout.flush()

    if _DEBUG_LOG:
        print('[DEBUG] -----')
        print('[DEBUG] mentions in context sentences:')
        print(c_corefs)
        print('[DEBUG] **********')
        sys.stdout.flush()

    ctx_mentions = get_mentions(c_corefs)
    if _MENTION_LOG:
        print(' * [DEBUG-MENTION] context sentence mentions:')
        for set_id in ctx_mentions.keys():
            for mention in ctx_mentions[set_id]:
                mention_txt = get_mention_txt(c_tt, mention)
                print(' * [DEBUG-MENTION] - Set {}: {}'.format(set_id, mention_txt))
                sys.stdout.flush()
        print(' * [DEBUG-MENTION] ----------')
        sys.stdout.flush()

    if _DEBUG_LOG:
        print('[DEBUG] ===============')
        sys.stdout.flush()

    use_avg_score = coreference_link_score == 'avg'
    metrics = []
    for cur_cid in cur_mentions.keys():
        if cur_cid in ctx_mentions:
            for cur_mnt in cur_mentions[cur_cid]:   
                # NOTE: from any mention with given set ID in current sentence, to any mention with given set ID in the context sentence.
                
                for ctx_mnt in ctx_mentions[cur_cid]:
                    # NOTE: a new metrics entry is added for each link from any mention in the current sentence to any mention in the context sentence
                    metrics.append( [True, False, False, 0.0] )

                    all_weights = []
                    all_lines_max_weight = 0.0
                    for i in cur_mnt:
                        cur_line_max_weight = max(att[i])
                        if cur_line_max_weight > all_lines_max_weight:
                            all_lines_max_weight = cur_line_max_weight
                        for j in ctx_mnt:
                            all_weights.append( att[i][j] )
                    link_score = compute_link_score( all_weights, avg=use_avg_score)
                    if link_score >= all_lines_max_weight:
                        metrics[-1][1] = True
                    metrics[-1][2] = sum(all_weights) > 0.0
                    metrics[-1][3] = link_score

    ############################
    return sys_s, sys_c, metrics
    ############################

    metrics = []
    cid = []
    cidx = []
    coref_links_by_ID = {}

    for sc in s_corefs:

        if len(cid) == 0 or cid[-1] != sc[1] or cidx[-1] != sc[0]-1: 
            metrics.append( [False, False, False, 0.0] )    # NOTE: first element is set_id match, only in this case this is an annotated gold coreference. See above for the other 3 values
        cid.append( sc[1] )
        cidx.append( sc[0] )

        max_att = 0.0
        n_att = 0
        for cc in c_corefs:
            if sc[1] == cc[1]:
                metrics[-1][0] = True
                metrics[-1][2] = metrics[-1][2] or att[sc[0]][cc[0]] != 0.0
                if use_avg_score:
                    max_att += att[sc[0]][cc[0]]
                    n_att += 1
                else:
                    if att[sc[0]][cc[0]] > max_att:
                        max_att = att[sc[0]][cc[0]]
                print('[DEBUG] COREF-LINK, found coreference link: id {}, tokens {} <-> {}, weight: {}'.format(sc[1], s_tt[sc[0]], c_tt[cc[0]], att[sc[0]][cc[0]]))
        if use_avg_score:
            annot_info.append( (sc[0], sc[1], max_att / n_att if n_att > 0 else 0.0) )
        else:
            annot_info.append( (sc[0], sc[1], max_att) )
        line_max = max(att[sc[0]])
        metrics[-1][1] = metrics[-1][1] or max_att >= line_max
        if max_att > metrics[-1][3]:
            metrics[-1][3] = max_att

    #metrics = [int(metrics[0]), int(metrics[1]), metrics[2]]
    for i in range(len(metrics)):
        metrics[i] = [int(metrics[i][0]), int(metrics[i][1]), int(metrics[i][2]), metrics[i][3]]
    for e in annot_info:
        if s_tt[e[0]][:2] != '#[':
            s_tt[e[0]] = '#[' + s_tt[e[0]] + ']#:' + e[1] + '-att:' + str(e[2])
    for e in c_corefs:
        if c_tt[e[0]][:2] != '#[':
            c_tt[e[0]] = '#[' + c_tt[e[0]] + ']#:' + e[1]

    print('[DEBUG] COREF-LINK, cur seq: {}'.format(' '.join(s_tt)))
    print('[DEBUG] COREF-LINK, ctx seq: {}'.format(' '.join(c_tt)))
    print('[DEBUG] COREF-LINK, metrics: {}'.format(metrics))

    return ' '.join(s_tt), ' '.join(c_tt), metrics

def analyze_and_evaluate(align_data, system_data, ctx_size, heads=None, seq_ids=None):
    """
    ***align_data***: system source/target reference data aligned to the ParCorFull2 corpus
    ***system_data***: system source/target data (generated data in case of target; reference data in case of source, but with a diffenret tokenization, and thus needing re-alignmeent)
    ***ctx_size***: num. of context sentences used by the model (usually 3 with Lorenzo's models)
    ***heads***: heads of document, that is index of the first sentence in each document => not used any more
    ***seq_ids***: IDs of sequences to process. If None all sequences are analyzed. This is used for example to select only the sentences of the set constructed by Dimitra (see the import in the main function).
    """

    def safe_clean(seq):
        res = seq.replace('&apos;', '\'')
        res = res.replace('&quot;', '"')
        res = res.replace('&#91;', '[')
        res = res.replace('&#93;', ']')
        res = res.replace('&amp;', '\&')

        return res.strip()

    def clean_for_passER(seq):

        res = safe_clean(seq)
        res = res.replace(' @-@ ', '-')
        res = res.replace('<END>', '')
        res = res.replace('<end>', '')
        res = res.replace('<pad>', '')
        res = res.replace('<eos>', '')
        res = res.replace('  ', ' ')
        res = res.replace('   ', ' ')

        return res.strip()
    
    # TODO: Copier le bloc suivant pour traiter full_ctx en dehors du traitement du contexte.
    # def something(ctx_s, ctx_seq, cur_seq, sys_cur_corefs, key, system_data):
    def something(ctx_s, ctx_seq, cur_seq, sys_cur_corefs, key, att, c2s_wer):
        """_summary_

        Args:
            ctx_s (_type_): _description_
            ctx_seq (_type_): _description_
            cur_seq (_type_): _description_
            sys_cur_corefs (_type_): _description_
            key (_type_): _description_
            system_data (_type_): _description_

        Returns:
            tuple: paire dont 
                0: le premier élément est la clé de la matrice d'attention
                1: 
        """
        er_vals = edit_distance.str_edit_distance(ctx_s[1], ctx_seq)
        c2s_wer += er_vals
        er_ctx_seq = clean_for_passER(ctx_seq)
        er_pass = edit_distance.str_edit_distance(ctx_s[1], er_ctx_seq)
        wer = er_pass.get_wer() # (nb_ins + nb_del + nb_sub) / len(snt)
        if er_pass.get_wer() >= wer_threshold:
            sys.stderr.write(' * FATAL ERROR: found too large divergence (WER: {:.2f}; Ins: {}, Del: {}, Sub: {}) between reference and system context sentence @{}\n'.format(er_pass.get_wer(), er_vals.nombre_erreur_insertion, er_vals.nombre_erreur_suppression, er_vals.nombre_erreur_substitution, key))
            sys.stderr.write(' *   Ref: {}\n'.format(ctx_s[1]))
            sys.stderr.write(' *   Sys: {}\n'.format(ctx_seq))
            sys.exit(0)
        a = er_vals.alignements # Alignement entre les phrases

        if _CORPUS_SYSTEM_COMPARISON_LOG:
            print(' * [DEBUG] ANALYSIS@{} ctx-gold: {}'.format(key, ctx_s[1]))
            print(' * [DEBUG] ANALYSIS@{} ctx-syst: {}'.format(key, ctx_seq))
            print(' * [DEBUG] **********')
            if _DEBUG_LOG:
                print(' * [DEBUG] ANALYSIS@{} context alignment: {}'.format(key, a))
            sys.stdout.flush()

        sys_ctx_corefs = find_coref_matches(ctx_s, ctx_seq, a)
        for scc in sys_ctx_corefs:
            token_identity_level.append( int(scc[2]) )
        # coref_res = find_coref_links(cur_seq, sys_cur_corefs, ctx_seq, sys_ctx_corefs, system_data[key]['att'])
        coref_res = find_coref_links(cur_seq, sys_cur_corefs, ctx_seq, sys_ctx_corefs, att)
        return (key, coref_res)

        
    c2s_wer = EditDistance(nombre_erreur_insertion=0, nombre_erreur_suppression=0, nombre_erreur_substitution=0, taille_tenseur_reference=0, alignements=[])

    offset = 0
    cur_bogus_idx = 0 if canmt_system == 'concat' else 1
    alignments = []
    analysis_results = []
    token_identity_level = []
    idx_old = 0
    k = 1 if canmt_system == 'concat' else -1
    heads = [186, 335, 457, 690, 792, 1034, 1268, 1420, 1610, 1755, 1782, 1814, 1863, 1910, 1932, 1958, 1989, 2009, 2044, 2075, 2108, 2123, 2153, 2191, 2214, 2245, 2262, 2280]
    for idx, src_s in enumerate(align_data):
        k = min(k+1, 3)
        #offset_idx = compute_head_offset(idx, src_heads)
        if idx >= 0: # and idx < 2127:
            if idx in heads:
                offset += 1
                k= -1
            # 1. Only for the 100 sentences subset
            if seq_ids is not None:
                if idx >= 186:
                    offset = 1
                if offset >= 336:
                    offset = 2
            # ------------------------------------
            cur_bogus_idx = k
            skip = False
            if seq_ids is None:
                key = str(offset+idx) + '-' + str(cur_bogus_idx)
                if key not in system_data:
                    skip = True
                else:
                    cur_seq = system_data[key]['cur']
                    if idx - idx_old > 1 or cur_seq == '<pad> <end>' or cur_seq == '<END>' or cur_seq == '( EN ) <END>':
                        # offset += 1
                        if _DEBUG_LOG:
                            print('[DEBUG] offset increased to {} at {}'.format(offset, idx))
                            sys.stdout.flush()
            else:
                found = False
                for bb_idx in range(1, ctx_size+1):
                    key = str(offset+idx) + '-' + str(bb_idx)
                    if key in system_data:
                        cur_bogus_idx = bb_idx
                        found = True
                        break
                if not found:
                    skip = True

            if _DEBUG_LOG:
                print('[DEBUG-KEY] skip: {}; offset, idx, cur_bogus_idx: {}, {}, {}'.format(skip, offset, idx, cur_bogus_idx))
                sys.stdout.flush()

            if not skip:
                offset_idx = offset + idx
                key = str(offset_idx) + '-' + str(cur_bogus_idx)
                
                # assert key in system_data, 'seq-id {} not found in system output data'.format(key)
                cur_seq = system_data[key]['cur']
                cur_seq = safe_clean(cur_seq)

                if _CORPUS_SYSTEM_COMPARISON_LOG:
                    print('[DEBUG] ANALYSIS corpus curr ({}): {}'.format(idx, src_s[1]))
                    print('[DEBUG] ANALYSIS system curr ({}): {}'.format(idx, cur_seq))
                    print('[DEBUG] **********')
                    sys.stdout.flush()

                er_vals = edit_distance.str_edit_distance(src_s[1], cur_seq)
                c2s_wer += er_vals
                er_cur_seq = clean_for_passER(cur_seq)
                er_pass = edit_distance.str_edit_distance(src_s[1], er_cur_seq)
                wer = er_pass.get_wer()
                if wer >= wer_threshold:
                    sys.stderr.write(' * FATAL ERROR: found too large divergence (WER: {:.3f}) between reference and system current sentence @{}\n'.format(wer, key))
                    sys.stderr.write(' *   Ref: {}\n'.format(src_s[1]))
                    sys.stderr.write(' *   Sys: {}\n'.format(er_cur_seq))
                    sys.exit(0)
                a = er_vals.alignements
                alignments.append( a )

                if _DEBUG_LOG:
                    print('[DEBUG] alignments: {}'.format(len(alignments)))
                    print('[DEBUG] ANALYSIS current alignment: {}'.format(a))
                    print('[DEBUG] ***************')
                    sys.stdout.flush()

                if True: #idx >= 3:
                    sys_cur_corefs = find_coref_matches(src_s, cur_seq, a)
                    for scc in sys_cur_corefs:
                        token_identity_level.append( int(scc[2]) )
                    if not _FULL_MATRICE:
                        for ctx_idx in range(1, ctx_size+1):
                            key = str(idx) + '-' + str(ctx_idx)
                            #assert key in system_data, 'seq-id {} not defined in system output data'.format(key)

                            ctx_seq_flag = system_data[key]['ctx'] != '<eos>' if key in system_data else False

                            if _DEBUG_LOG:
                                print('[DEBUG-KEY] ctx key defined: {}; ctx-seq-flag: {}; key: {} (offset_idx, ctx_idx: {}, {})'.format(key in system_data, ctx_seq_flag, key, offset_idx, ctx_idx))
                                sys.stdout.flush()

                            if key in system_data and ctx_seq_flag:
                                ctx_s = align_data[idx-ctx_idx]
                                ctx_seq = system_data[key]['ctx']
                                ctx_seq = safe_clean(ctx_seq)
                                if ctx_seq != '<end>':  # NOTE: in multi-enc system output, when the current sentence is at the begin of a document, context sentences are just '<end>' tokens
                                    # if _FULL_MATRICE:
                                    #     full_ctx_seq = " ".join([full_ctx_seq, ctx_seq]) # On concatene les phrases en ajoutant un caractère espace entre
                                    #     full_att = torch.cat([full_att, system_data[key]['att']], dim = 1)
                                    #     for i in range(len(ctx_s)):
                                    #         full_ctx_s[i] += ctx_s[i]
                                    # else:
                                        analysis_results.append(something(ctx_s, ctx_seq, cur_seq, sys_cur_corefs, key, system_data[key]['att'], c2s_wer) )
                    if _FULL_MATRICE : # Si on étudie l'attention crt x [k3,k2,k1] alors on doit fusionner les éléments
                        ctx_seq = system_data[key]['ctx']
                        ctx_seq = safe_clean(ctx_seq)
                        if ctx_seq != '<end>':
                            ctx_s = ['', '', [], [], ''] 
                            full_ctx_s = ['', '', [], [], ''] 
                            full_ctx_seq = "" 
                            system_data_key = ""
                            # full_att = torch.zeros((len(system_data[key]['att']), 0)) 
                            # for ctx_idx in range(ctx_size, 0, -1): # Pour chaque contexte possible (parcours inversé pour garder l'ordre de lecture)
                            limit = False
                            for ctx_idx in range(1, ctx_size + 1):
                                key = str(idx + offset) + '-' + str(ctx_idx)
                                if not limit:
                                    ctx_s = align_data[idx-ctx_idx] if (idx-ctx_idx >= 0) else ['', '', [], [], '']
                                    for i in range(len(ctx_s)):
                                        full_ctx_s[i] = ctx_s[i] + full_ctx_s[i]

                                # key = str(idx) + '-' + str(ctx_idx)
                                if key in system_data and \
                                    (system_data[key]['ctx'] != '<eos>'): # or system_data[key]['ctx'] != '<end>') : 
                                    # Si le fichier existe pour crt x k alors
                                    full_ctx_seq = safe_clean(system_data[key]['ctx']) 
                                    # On vérifie que le ctx n'est pas vide (<=> vérifier si != '<eos>)
                                    # On concatene les phrases en ajoutant un caractère espace entre
                                    limit = True
                                    system_data_key = key
                                    # On concatene les matrices d'attentions sur les colonnes 
                                    # full_att = torch.cat([torch.Tensor(system_data[key]['att']), full_att], dim = 1)
                                    # On concatène tout les éléments de la structure ctx_s
                                    
                                    
                            if full_ctx_seq.split() != [] and len(ctx_s) > 0:
                                idx_old = idx
                                analysis_results.append(something(full_ctx_s, full_ctx_seq, cur_seq, sys_cur_corefs, key, system_data[system_data_key]['att'], c2s_wer))
    if _DEBUG_LOG and c2s_wer.taille_tenseur_reference > 0:
        print('[DEBUG-KEY] corpus vs. system sentences WER: {:.2f}'.format( c2s_wer.get_wer() ))
        sys.stdout.flush()

    return analysis_results, token_identity_level

def main(args):

    from subset_ids import dimitra_ids, ctx_needed_concat_ids, ctx_needed_and_hard_coref_concat_ids, pos_att_concat_ids
    import os
    #subset_ids=dimitra_ids # NOTE: modify the value of this to constrain the sentence ID set (e.g. dimitra_ids corresponds to the subset identified by Dimitra Niaouri, M2 internship in 2022; see TAL 2024 journal paper).
    if _CTX_NEEDED_AND_HARD_COREF_CONCAT_IDS:
        subset_ids=ctx_needed_and_hard_coref_concat_ids
    else:
        subset_ids=None

    ctx_size = 3
    src = read_txt(args.corpus_source)
    tgt = read_txt(args.corpus_target)
    src_heads_file = '/'.join(args.corpus_source.split('/')[:-1]) + '/test.en-de.en.heads'
    tgt_heads_file = '/'.join(args.corpus_target.split('/')[:-1]) + '/test.en-de.de.heads'
    src_heads = read_txt(src_heads_file)
    src_heads = [int(i) for i in src_heads]
    tgt_heads = read_txt(tgt_heads_file)
    tgt_heads = [int(i) for i in tgt_heads]

    system_src_list = args.system_data

    print(' * Read {} source sentences'.format(len(src)))
    print(' * Read {} target sentences'.format(len(tgt)))
    print(' ***')
    sys.stdout.flush()

    # disco_src_data, disco_tgt_data = read_discomt_data()
    # print("[debug] écriture des données")
    # with open('/home/getalp/lopezfab/Bureau/Attention_git/Tests/disco_tgt_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(disco_tgt_data, f)
    # print('[debug] disco_tgt_data écrit')
    # Check if the JSON files exist
    
    disco_src_data_path = f'{_PATH}/Tests/disco_src_data.json'
    disco_tgt_data_path = f'{_PATH}/Tests/disco_tgt_data.json'
    news_src_data_path = f'{_PATH}/Tests/news_src_data.json'
    news_tgt_data_path = f'{_PATH}/Tests/news_tgt_data.json'
    aligned_src_data_path = f'{_PATH}/Tests/aligned_src_data.json'
    aligned_tgt_data_path = f'{_PATH}/Tests/aligned_tgt_data.json'

    if os.path.exists(disco_src_data_path) and os.path.exists(disco_tgt_data_path):
        with open(disco_src_data_path, 'r', encoding='utf-8') as f:
            disco_src_data = json.load(f)
        with open(disco_tgt_data_path, 'r', encoding='utf-8') as f:
            disco_tgt_data = json.load(f)
        print('[debug] disco_src_data_path and disco_tgt_data loaded from json file')
    else:
        disco_src_data, disco_tgt_data = read_discomt_data()
        print('[debug] disco_src_data_path and disco_tgt_data extract from text file')


    if os.path.exists(news_src_data_path) and os.path.exists(news_tgt_data_path):
        with open(news_src_data_path, 'r', encoding='utf-8') as f:
            news_src_data = json.load(f)
        with open(news_tgt_data_path, 'r', encoding='utf-8') as f:
            news_tgt_data = json.load(f)
        print('[debug] news_src_data and news_tgt_data loaded from json file')
    else:
        news_src_data, news_tgt_data = read_news_data()
        print('[debug] news_src_data and news_tgt_data extract from text file')
    

    print(' * Read {} raw text for DiscoMT data'.format(len(disco_src_data['text'].keys())), flush=True)

    # print('[debug] écriture des données news')
    # news_src_data, news_tgt_data = read_news_data()
    # with open('/home/getalp/lopezfab/Bureau/Attention_git/Tests/news_src_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(news_src_data, f)
    # with open('/home/getalp/lopezfab/Bureau/Attention_git/Tests/news_tgt_data.json', 'w', encoding='utf-8') as f:
    #     json.dump(news_tgt_data, f)
    # print('[debug] news_src_data et news_tgt_data écrit')


    print(' * Read {} raw text for news data'.format(len(news_src_data['text'].keys())), flush=True)
   
    if os.path.exists(aligned_src_data_path) and os.path.exists(aligned_tgt_data_path):
        with open(aligned_src_data_path, 'r', encoding='utf-8') as f:
            aligned_src_data_path = json.load(f)
        with open(aligned_tgt_data_path, 'r', encoding='utf-8') as f:
            aligned_tgt_data_path = json.load(f)
        print('[debug] aligned_src_data_path and aligned_tgt_data_path loaded from json file')
    else:
        aligned_src = match_nmt2parcorfull(src, disco_src_data, news_src_data)  # TODO: modify the returned struct to be a dictionary or a NamedTuple like the EncoderOut structure
        print(' *** source side aligned to ParCorFull2 ***', flush=True)
        aligned_tgt = match_nmt2parcorfull(tgt, disco_tgt_data, news_tgt_data)
        print(' *** target side aligned to ParCorFull2 ***', flush=True)
    

    assert len(aligned_src) == len(aligned_tgt)

    system_src_data = read_system_src_data( system_src_list, seq_ids=subset_ids )

    print(' * Read {} system sentences'.format(len(system_src_data)), flush=True)

    analyzed_ref = aligned_src
    analyzed_hyp = system_src_data
    if eval_language == 'target':
        analyzed_ref = aligned_tgt 
    analysis_results, til = analyze_and_evaluate(analyzed_ref, analyzed_hyp, ctx_size, heads=None, seq_ids=subset_ids)
    if _VERBOSE:
        for src_s, tgt_s in zip(aligned_src, aligned_tgt):
            print('Src -->: {}'.format(src_s[1]))
            print(' +Coref: {}'.format(src_s[-1]))
            print(' -----')
            print('Tgt -->: {}'.format(tgt_s[1]))
            print(' +Coref: {}'.format(tgt_s[-1]))
            print(' ************************************************** ')
        sys.stdout.flush()

    # output_file = sys.argv[3] + '.results'
    f = open(output_file, 'w', encoding='utf-8')
    metrics = [0, 0, 0, 0]
    for e in analysis_results:
        fid = e[0]
        res = e[1]
        for m in res[2]:
            metrics[0] += m[0]
            metrics[1] += m[1]
            metrics[2] += m[2]
            metrics[3] += m[3]
        f.write(fid + '\n')
        f.write('current: ' + res[0] + '\n')
        f.write('context: ' + res[1] + '\n')
        f.write(' -----\n')
    f.close()

    if metrics[0] > 0:
        print(' ********** ')
        print(' Analysis results written in {}'.format(output_file))
        try:
            print(f" * Analysis done on {args.system_data}")
        except:
            print(f"wrong argument for args.system_data")
        print(' -----')
        print(' * Evaluation:')
        print(' * Corpus-to-system mention token identity level: {:.2f}%'.format( (sum(til)/len(til))*100.0 if len(til) > 0 else 0.0))
        num_ex = metrics[0]
        print(' * Max weight metric: {} / {} = {:.2f}'.format(metrics[1], num_ex, metrics[1] / num_ex *100.0))
        print(' * Non-zero weight metric: {} / {} = {:.2f}'.format(metrics[2], num_ex, metrics[2] / num_ex *100.0))
        print(' * Average weight metric: {} / {} = {:.4f}'.format(metrics[3], num_ex, metrics[3] / num_ex))
        print(' **********')
    else:
        print(' *** NO MATCH !!!')
        print(f'{metrics}')

main(args)

