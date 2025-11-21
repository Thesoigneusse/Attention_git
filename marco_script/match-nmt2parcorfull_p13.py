import re
import json
import sys
from typing import List
from typing import Tuple

import torch
import argparse
import xml.etree.cElementTree as ET

from pathlib import Path
from typing import List, Dict, TypedDict, Any
from Classes_p13 import edit_distance
from Classes_p13.EditDistance import EditDistance
from Classes_p13.SentenceAlignement import SentenceAlignement
from Classes_p13.WordAlignement import WordAlignement
from Classes_p13.CoreferenceMatch import CoreferenceMatch
from Classes_p13.SystemOutput import SystemOutput
from Classes_p13.WeightAnalysisMetric import WeightAnalysisMetric
from Classes_p13.Coref import Coref
from Classes_p13.Data import Data
# Activate the following python environment for importing the German BERT:
# source /home/getalp/dinarelm/anaconda3/bin/activate ssl_wav2vec2_torch18

#from transformers import AutoModel, AutoTokenizer
#tokenizer = AutoTokenizer.from_pretrained('dbmdz/bert-base-german-cased')
#model = AutoModel.from_pretrained('dbmdz/bert-base-german-cased')

# Exemple d'appel pour le modèle concat:
# python3 ./marco_script/match-nmt2parcorfull.py \
#    /home/getalp/lopezfab/Attention/k3/GOLD/test.en \
#    /home/getalp/lopezfab/Attention/k3/GOLD/test.de \
#    ./marco_script/list_filename_matrice/concat/full_matrice/list_chemin_1_3.txt \
#    --output-file=./Output/debug_metrics.results \
#    --canmt-system=concat \
#    --pudb=False --local=False

# Exemple d'appel pour le modèle multi_enc:
# python3 ./marco_script/match-nmt2parcorfull.py \
#    /home/getalp/lopezfab/Attention/k3/GOLD/test.en \
#    /home/getalp/lopezfab/Attention/k3/GOLD/test.de \
#    ./marco_script/list_filename_matrice/multi_enc/full_matrice/list_chemin_1_3.txt \
#    --output-file=./Output/debug_metrics.results \
#    --canmt-system=multienc \
#    --pudb=False --local=False

_VERBOSE = True
_CORPUS_SYSTEM_COREF_MATCHES = False
_CORPUS_SYSTEM_COMPARISON_LOG = False
_MENTION_LOG = False
_DEBUG_LOG = True

# Par convention, dans le cas d'une _FULL_MATRICE, le numero de contexte est le contexte le plus éloigné de la phrase courante
_FULL_MATRICE = True 

_DEBUG = True
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
parser.add_argument('--local', type=str, default="False", help='Specify if the script is run on the serveurs or not. Default: False')
parser.add_argument('--reload-data', type=str, default="False", help='If the model have to reprocess the data. Default: False')
parser.add_argument('--reload-align-data', type=str, default="False", help='If the model have to reprocess the alignement process. Default: False')

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

_LOCAL: bool = False if args.local == "False" else True
_PATH: Path= Path("/home/getalp/lopezfab/lig/Attention_git") if _LOCAL else Path("/home/getalp/lopezfab/Attention_git")
_DATAPATH: Path = _PATH / "marco_script/data"
_RELOAD_DATA: bool = True if args.reload_data == 'True' else False
_RELOAD_ALIGN_DATA: bool = True if args.reload_align_data == 'True' else False



def read_txt(filename: Path) -> List[str]:
    """
    Read lines from a text file and return them as a list of stripped strings.
    Args:
        filename (Path): The path to the text file to read.
    Returns:
        List[str]: A list of strings, where each string is a line from the file
                   with leading and trailing whitespace removed.
    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: If an error occurs while reading the file.    
    """
    with filename.open(mode="r", encoding='utf-8') as f: 
        lines = f.readlines()
    return [line.strip() for line in lines]

def pref2raw(prefix: str) -> str:
    """Permet de découper les noms des fichiers"""
    tt = prefix.split('_')
    assert len(tt) == 2
    raw = '00' + tt[1]
    if len(raw) == 5:
        raw = '0' + raw
    return raw

def en_tokenize(text: str) -> str:
    """
    Ad-hoc tokenization to have raw and tokenized sentences in the corpus match. 
    Not beautifull but shoult work
    """

    # punctuation: List[str] = ['.', ',', ';', ':', '!', '?', '"', '\'', '[', ']'] 
    # Liste des éléments de ponctuation
    stripped_text = text.strip()

    # Delete punction token
    tok_str = ' '.join(re.split(r'(\.\.\.|\.|,|;|:|\!|\?|"|\'|\[|\]|„|“|‚|‘|”|«|»|\(|\))', stripped_text))

    token_to_replace: Dict[str, str] = {
            ' \' s ' : ' \'s ',
            ' \' d ' : ' \'d ',
            ' \' m ' : ' \'m ',
            'n \' t ' : ' n\'t ',
            ' \' re ' : ' \'re ',
            ' \' ve ' : ' \'ve ',
            ' \' ll ' : ' \'ll ',
            ' \' 60s ' : ' \'60s ',
            ' \' 70s ' : ' \'70s ',
            ' \' 80s ' : ' \'80s ',
            ' \' 90s ' : ' \'90s ',
            ' D . S ' : ' D.S ',
            ' U . S . ' : ' U.S. ',
            ' D . C . ' : ' D.C. ',
            ' o \' clock' : ' o\'clock',
            ' Dr . ' : ' Dr. ',
            ' mid- \'90s ' : ' mid-\'90s ',
            ' I . P ' : ' I.P ',
            ' P . C ' : ' P.C ',
            ' Amazon . com' : ' Amazon.com',
            ' cannot' : ' can not',
            ' U . K .' : ' U.K.',
            ' Ph . D . ' : ' Ph.D. ',
            ' wanna' : ' wan na',
            ' E . T . ' : ' E.T. ',
    }
    for token in token_to_replace:
        tok_str.replace(token, token_to_replace[token])
    tok_str = re.sub(r' (\d+) , (\d\d\d)', ' \\1,\\2', tok_str )
    tok_str = re.sub(r'(\d+) . (\d+)', '\\1.\\2', tok_str)
    tok_str = re.sub(r' \' (\d\d) ', ' \'\\1 ', tok_str)
    tok_str = re.sub(r' \$(\d+) ', ' $ \\1 ', tok_str)
    # Ad hoc processing for a specific sentence:
    if 'unbelievable movie about what' in tok_str:
        splitted_token: List[str] = tok_str.split()
        new_str: List[str] = []
        first: bool = True
        for token in splitted_token:
            if token == 'E.T.' and first:
                new_str.append(token)
                first = False
            elif token == 'E.T.':
                new_str.append('E.T')
                new_str.append('.')
            else:
                new_str.append(token)
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

def de_tokenize(text: str) -> str:
    """
    Ad-hoc tokenization to have raw and tokenized sentences in the corpus match.
    Not beautiful but should work
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

def get_words(filename: Path, key_prefix: str) -> Dict[str, str | None]:
    """
    Parse an XML file and extract words indexed by their IDs.

    Each `<word>` element in the XML file is expected to have an "id" attribute
    and optional text content. The function builds a dictionary mapping a key
    composed of the given prefix and the word ID (formatted as
    "<key_prefix>-<id>") to the word's text.

    Args:
        filename (Path): Path to the XML file to parse.
        key_prefix (str): Prefix added before each word ID to build the dictionary key.

    Returns:
        Dict [str, str | None]: A dictionary mapping prefixed word IDs to their text content.
                               The value is `None` if the XML element has no text.
    """
    words: Dict[str, str | None] = {}
    for word in ET.parse(filename).getroot():
        key = key_prefix + '-' + word.attrib['id']
        words[key] = word.text
    return words

def get_span_idx(span_str: str) -> List[int]:
    """
    Extract word indices from a span string.

    A span string typically contains patterns such as "word_1..word_3" or
    "word_5", and this function returns the corresponding list of integer indices.

    Examples:
        >>> get_span_idx("word_5")
        [5]
        >>> get_span_idx("word_2..word_4")
        [2, 3, 4]

    Args:
        span_str (str): A string containing word index markers (e.g., "word_1..word_3").

    Returns:
        List[int]: A list of word indices extracted from the span.
    """
    indices: List[int] = []

    # Split the string around '..' or whitespace
    parts = span_str.replace('..', ' ').split()

    for token in parts:
        if token.startswith('word_'):
            try:
                indices.append(int(token[5:]))
            except ValueError:
                raise ValueError(f"Invalid word index in token: '{token}'")

    if len(indices) == 2:
        # Generate the full inclusive range between the two indices
        start, end = indices
        if start > end:
            raise ValueError(f"Invalid span order: start ({start}) > end ({end}) in '{span_str}'")
        indices = list(range(start, end + 1))
    elif len(indices) > 2:
        raise ValueError(f"Unexpected number of indices ({len(indices)}) in span: '{span_str}'")

    return indices

def get_all_spans(span_str: str) -> List[List[int]]:
    """
    Extract all word index spans from a string.

    Each span is separated by commas or whitespace and may include single or
    range-like spans such as "word_1..word_3". The function delegates to
    `get_span_idx()` to parse each individual span.

    Examples:
        >>> get_all_spans("word_1..word_3, word_5")
        [[1, 2, 3], [5]]
        >>> get_all_spans("word_10")
        [[10]]

    Args:
        span_str (str): A string containing one or more spans separated by commas or spaces.

    Returns:
        List[List[int]]: A list of spans, where each span is a list of word indices.
    """
    spans: List[List[int]] = []

    # Split spans by comma or whitespace, ignoring empty tokens
    for part in span_str.replace(',', ' ').split():
        if not part.strip():
            continue
        indices = get_span_idx(part)
        spans.append(indices)

    return spans

def get_coreferences(
    filename: Path,
    key_prefix: str
) -> Tuple[List[Coref], Dict[str, str]]:
    """
    Extrait les mentions de co-référence d'un fichier XML.

    Chaque mention contient :
        - 'id': identifiant de la mention
        - 'fileid': identifiant du fichier (préfixe)
        - 'span': chaîne décrivant les indices de tokens concernés
        - 'coref_class': classe de co-référence
        - 'mention': texte de la mention

    La fonction retourne également un dictionnaire qui mappe chaque token
    (clé formée par "<key_prefix>-word_<index>") à la classe de co-référence correspondante.

    Args:
        filename (Path): Chemin vers le fichier XML contenant les mentions.
        key_prefix (str): Préfixe à utiliser pour construire les clés des mots.

    Returns:
        Tuple[List[Dict[str, str]], Dict[str, str]]:
            - Liste des mentions extraites.
            - Dictionnaire mapant les mots à leur classe de co-référence.
    """
    if _DEBUG:
        print(f'[DEBUG] get_coreferences, extracting mentions from file {filename}')

    corefs: List[Coref] = []
    words_in_corefs: Dict[str, str] = {}

    tree = ET.parse(filename)
    root: ET.Element[str] = tree.getroot()

    for markable in root:
        mention_id: str = markable.attrib['id']
        span_str: str = markable.attrib['span']
        coref_class: str = markable.attrib['coref_class']
        mention_text: str = markable.attrib.get('mention', '')

        # Ajout de la mention dans la liste
        corefs.append({
            'id': mention_id,
            'fileid': key_prefix,
            'span': span_str,
            'coref_class': coref_class,
            'mention': mention_text
        })

        # Conversion de la chaîne de span en indices
        spans = get_all_spans(span_str)
        for span in spans:
            for idx in span:
                key = f"{key_prefix}-word_{idx}"
                words_in_corefs[key] = coref_class

                if _DEBUG:
                    print(f'[DEBUG] get_coreferences: adding words_in_corefs with key {key}')

    return corefs, words_in_corefs

def read_discomt_data() -> Tuple[Data, Data]:
    """
    Load and process the DiscoMT corpus, including raw text, tokenized text,
    word mappings, and coreference annotations for both source (EN) and target (DE) data.

    Returns two dictionaries:
    - `src_data`: source-side corpus data
    - `tgt_data`: target-side corpus data

    Each dictionary contains:
        - "text": mapping from raw sentences to (tokenized sentences, alignments)
        - "words": mapping from unique word IDs to tokenized words
        - "coref": list of dictionaries, each describing a coreference mention
        - "words_in_coref": mapping from unique word IDs to entity IDs

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]: (src_data, tgt_data)
    """
    base_path = Path(f"{_DATAPATH}/ParCorFull2/parcor-full/corpus/DiscoMT")
    src_path: Path = base_path / "EN"
    tgt_path: Path = base_path / "DE"

    prefixes = [
        "000_1756", "001_1819", "002_1825", "003_1894", "005_1938",
        "006_1950", "007_1953", "009_2043", "010_205", "011_2053"
    ]

    raw_txt_path: Path = Path("Source/sentence")
    word_path: Path = Path("Basedata")
    markable_path: Path = Path("Markables")

    src_wer: EditDistance = EditDistance(0, 0, 0, 0, [])
    tgt_wer: EditDistance = EditDistance(0, 0, 0, 0, [])

    src_raw2tok_text: Dict[str, Tuple[str, List[WordAlignement]]] = {}
    tgt_raw2tok_text: Dict[str, Tuple[str, List[WordAlignement]]] = {}

    # === Step 1: Read and tokenize raw text ===
    for prefix in prefixes:
        raw_id: str = pref2raw(prefix)

        src_file: Path = src_path / raw_txt_path / f"talk{raw_id}.de-en.en"
        tgt_file: Path = tgt_path / raw_txt_path / f"talk{raw_id}.de-en.de"

        with src_file.open(encoding="utf-8") as f_src, tgt_file.open(encoding="utf-8") as f_tgt:
            src_lines: List[str] = f_src.readlines()
            tgt_lines: List[str] = f_tgt.readlines()

        if len(src_lines) != len(tgt_lines):
            raise ValueError(f"Line mismatch in prefix {prefix}: {len(src_lines)} vs {len(tgt_lines)}")

        for raw_src, raw_tgt in zip(src_lines, tgt_lines):
            if not raw_src.strip() or not raw_tgt.strip():
                continue

            # Source-side alignment
            tok_src: str = en_tokenize(raw_src)
            er_src: EditDistance = edit_distance.str_edit_distance(tok_src, raw_src)
            src_wer += er_src
            if len(er_src.alignements) != len(tok_src.split()):
                raise ValueError(
                    f"Source alignment mismatch ({len(er_src.alignements)} vs {len(tok_src.split())})"
                )
            src_raw2tok_text[raw_src.strip()] = (tok_src, er_src.alignements)

            # Target-side alignment 
            tok_tgt = de_tokenize(raw_tgt)
            er_tgt = edit_distance.str_edit_distance(tok_tgt, raw_tgt)
            tgt_wer += er_tgt
            if len(er_tgt.alignements) != len(tok_tgt.split()):
                raise ValueError(
                    f"Target alignment mismatch ({len(er_tgt.alignements)} vs {len(tok_tgt.split())})"
                )
            tgt_raw2tok_text[raw_tgt.strip()] = (tok_tgt, er_tgt.alignements)

    # src_data: Dict[str, Dict[str, Tuple[str, List[WordAlignement]]]] = {"text": src_raw2tok_text}
    # tgt_data: Dict[str, Dict[str, Tuple[str, List[WordAlignement]]]] = {"text": tgt_raw2tok_text}
    src_data: Data = {
        "text": {},
        "words": {},
        "coref": [],
        "words_in_coref": {}
    }

    tgt_data: Data = {
        "text": {},
        "words": {},
        "coref": [],
        "words_in_coref": {}
    }
    src_data['text'] = src_raw2tok_text
    tgt_data['text'] = tgt_raw2tok_text


    if _DEBUG_WER:
        print(f"[DEBUG-WER] DiscoMT source-side WER: {src_wer.get_wer() * 100:.2f}%")
        print(f"[DEBUG-WER] DiscoMT target-side WER: {tgt_wer.get_wer() * 100:.2f}%")
        sys.stdout.flush()

    # === Step 2: Read word XML files ===
    src_words: Dict[str, str | None] = {}
    tgt_words: Dict[str, str | None] = {}

    for prefix in prefixes:
        src_words.update(get_words(src_path / word_path / f"{prefix}_words.xml", prefix))
        tgt_words.update(get_words(tgt_path / word_path / f"{prefix}_words.xml", prefix))

    src_data["words"] = src_words
    tgt_data["words"] = tgt_words

    # === Step 3: Read coreference annotations ===
    src_coref: List[Coref] = []
    tgt_coref: List[Coref] = []
    src_words_in_coref: Dict[str, str] = {}
    tgt_words_in_coref: Dict[str, str] = {}

    for prefix in prefixes:
        src_coref_file = src_path / markable_path / f"{prefix}_coref_level.xml"
        tgt_coref_file = tgt_path / markable_path / f"{prefix}_coref_level.xml"

        src_coref_entries, src_words_map = get_coreferences(src_coref_file, prefix)
        tgt_coref_entries, tgt_words_map = get_coreferences(tgt_coref_file, prefix)

        src_coref.extend(src_coref_entries)
        tgt_coref.extend(tgt_coref_entries)
        src_words_in_coref.update(src_words_map)
        tgt_words_in_coref.update(tgt_words_map)

    src_data: Data = {
        "text": {},
        "words": {},
        "coref": [],
        "words_in_coref": {}
    }

    tgt_data: Data = {
        "text": {},
        "words": {},
        "coref": [],
        "words_in_coref": {}
    }

    src_data["coref"] = src_coref
    tgt_data["coref"] = tgt_coref
    src_data["words_in_coref"] = src_words_in_coref
    tgt_data["words_in_coref"] = tgt_words_in_coref

    return src_data, tgt_data

def read_news_data() -> Tuple[Data, Data]:
    """
    Lit les données « news » du corpus ParCorFull 2 (EN → DE) et produit :

        - Données source (EN)
        - Données cible (DE)

    La fonction extrait :
        1. Le texte brut et sa version tokenisée + alignements (raw → tok)
        2. Les mots annotés (fichiers *_words.xml)
        3. Les co-références (fichiers *_coref_level.xml)

    Returns:
        Tuple[Dict, Dict]:
            - src_data : données côté source (EN)
            - tgt_data : données côté cible (DE)
    """

    base_path = Path(_DATAPATH) / "ParCorFull2/parcor-full/corpus/news/"
    src_path = base_path / "EN"
    tgt_path = base_path / "DE"

    prefixes = [
        "03", "04", "05", "07", "08", "09", "10",
        "13", "16", "17", "18", "19", "20", "21",
        "22", "23", "24", "25"
    ]

    raw_dir = "Source"
    word_dir = "Basedata"
    markable_dir = "Markables"

    # --- 1. Mesure du WER raw → tokenized ---
    src_wer = EditDistance(0, 0, 0, 0, [])
    tgt_wer = EditDistance(0, 0, 0, 0, [])

    src_raw2tok: Dict[str, Tuple[str, List[WordAlignement]]] = {}
    tgt_raw2tok: Dict[str, Tuple[str, List[WordAlignement]]] = {}

    src_sentences: List[str] = []
    tgt_sentences: List[str] = []

    # --- 1a. Lecture du texte brut ---
    for prefix in prefixes:
        # Source EN
        src_file = src_path / raw_dir / f"{prefix}.en.xml"
        # NOTE: Strange behaviour copié de _PATH : paragraph ne peut etre décomposé avec *for*
        #       L.425 dans match-nmt2parcorfull.py
        #       -> seg n'est pas censé exister
        for paragraph in ET.parse(src_file).getroot():
            paragraph: ET.Element[str]
            for seg in paragraph:
                src_sentences.append(seg.text.strip()) # type: ignore
        

        # Target DE
        tgt_file = tgt_path / raw_dir / f"{prefix}.de.xml"
        # NOTE: Strange behaviour copié de _PATH : paragraph ne peut etre décomposé avec *for*
        #       -> seg n'est pas censé exister
        for paragraph in ET.parse(tgt_file).getroot():
            for seg in paragraph:
                tgt_sentences.append(seg.text.strip()) # type: ignore

    assert len(src_sentences) == len(tgt_sentences), "Mismatch EN/DE sentence count"

    # --- 1b. Tokenisation et alignement ---
    for s_en, s_de in zip(src_sentences, tgt_sentences):
        # Source EN
        tok_en = en_tokenize(s_en)
        ed_en = edit_distance.str_edit_distance(tok_en, s_en)
        src_wer += ed_en
        src_raw2tok[s_en] = (tok_en, ed_en.alignements)

        # Target DE
        tok_de = de_tokenize(s_de)
        ed_de = edit_distance.str_edit_distance(tok_de, s_de)
        tgt_wer += ed_de
        tgt_raw2tok[s_de] = (tok_de, ed_de.alignements)

    if _DEBUG_WER:
        print(f"[DEBUG-WER] EN raw→tok WER: {src_wer.get_wer()*100:.2f}")
        print(f"[DEBUG-WER] DE raw→tok WER: {tgt_wer.get_wer()*100:.2f}")

    # --- 2. Lecture des words.xml ---
    src_words: Dict[str, str | None] = {}
    tgt_words: Dict[str, str | None] = {}

    for prefix in prefixes:
        src_words.update(get_words(src_path / word_dir / f"{prefix}_words.xml", prefix))
        tgt_words.update(get_words(tgt_path / word_dir / f"{prefix}_words.xml", prefix))

    # --- 3. Lecture des co-références ---
    src_coref: List[Coref] = []
    tgt_coref: List[Coref] = []

    src_words_in_coref: Dict[str, str] = {}
    tgt_words_in_coref: Dict[str, str] = {}

    for prefix in prefixes:
        # EN
        coref_file: Path = src_path / markable_dir / f"{prefix}_coref_level.xml"
        coref_list: List[Coref]
        word_map: Dict[str, str]
        coref_list, word_map = get_coreferences(coref_file, prefix)
        src_coref.extend(coref_list)
        src_words_in_coref.update(word_map)

        # DE
        coref_file = tgt_path / markable_dir / f"{prefix}_coref_level.xml"
        coref_list, word_map = get_coreferences(coref_file, prefix)
        tgt_coref.extend(coref_list)
        tgt_words_in_coref.update(word_map)

    # Assemblage final
    src_data: Data = {
        "text": src_raw2tok,
        "words": src_words,
        "coref": src_coref,
        "words_in_coref": src_words_in_coref,
    }

    tgt_data: Data = {
        "text": tgt_raw2tok,
        "words": tgt_words,
        "coref": tgt_coref,
        "words_in_coref": tgt_words_in_coref,
    }

    return src_data, tgt_data

def annotate_coref_sentence(word_ids: List[str], data: Data, debug: bool = False) -> str:
    """Return the sentence with coreference decorations.

    """
    annotated: List[str] = []

    for word_id in word_ids:
        if word_id in data["words_in_coref"]:
            mention_id = data["words_in_coref"][word_id]
            annotated.append(f"#[{data['words'][word_id]}]#-{mention_id}")
        else:
            annotated.append(data["words"][word_id]) # type:ignore

    if debug:
        coref_count = sum(wid in data["words_in_coref"] for wid in word_ids)
        print(f"[DEBUG] Coreferent words: {coref_count}/{len(word_ids)}")

    return " ".join(annotated)

def match_nmt2parcorfull(
    src_nmt: List[str],
    disco_src_pcf: Tuple[Data, Data],
    news_src_pcf: Tuple[Data, Data],
    verbose: bool = False,
    debug: bool = False
) -> List[List[Any]]:
    """
    Match NMT reference sentences with their corresponding source sentences
    in the ParCorFull 2 corpus (DiscoMT or News).

    Parameters
    ----------
    src_nmt : List[str]
        Sentences used as input by the NMT system.
    disco_src_pcf : Dict[str, Any]
        Parsed ParCorFull 2 DiscoMT source corpus.
        Must contain 'text', 'words', and 'words_in_coref'.
    news_src_pcf : Dict[str, Any]
        Parsed ParCorFull 2 News source corpus.
        Same structure as `disco_src_pcf`.
    verbose : bool, optional
        Print match/mismatch information.
    debug : bool, optional
        Enable detailed debugging logs.

    Returns
    -------
    List[List[Any]]
        For each matched sentence, a structure containing:
        1. NMT input sentence (string)
        2. Tokenized corpus sentence (string)
        3. List of unique token IDs (List[str])
        4. Word alignments (original tuple in corpus)
        5. Annotated sentence with coreferent mentions (string)

    Raises
    ------
    ValueError
        When a token mismatch is encountered or if a sentence
        cannot be found in either corpus.
    """

    def annotate_coref_sentence(wids: List[str], data: Data) -> str:
        """Return the sentence with coreference decorations.
        """
        annotated: List[str] = []

        for wid in wids:
            if wid in data["words_in_coref"]:
                mention_id = data["words_in_coref"][wid]
                annotated.append(f"#[{data['words'][wid]}]#-{mention_id}")
            else:
                annotated.append(data["words"][wid]) # type:ignore

        if debug:
            coref_count = sum(wid in data["words_in_coref"] for wid in wids)
            print(f"[DEBUG] Coreferent words: {coref_count}/{len(wids)}")

        return " ".join(annotated)

    def match_sentence_in_corpus(
        sentence: str,
        text_dict: Dict[str, Tuple[str, Any]],
        words_list: List[Tuple[str, str]],
        word_idx: int,
        corpus_name: str
    ) -> Tuple[List[Any] | None, int]:
        """
        Attempt to match a tokenized NMT sentence against a linear word list from a
        ParCorFull2 corpus. Returns the structured match and the updated word index.

        Parameters
        ----------
        sentence : str
            Raw NMT sentence to match.

        text_dict : Dict[str, Tuple[str, Any]]
            Mapping:
                raw_sentence -> (tokenized_sentence, alignment_structure)

        words_list : List[Tuple[str, str]]
            List of (word_id, token) pairs from the corpus, in corpus order.

        word_idx : int
            Current index in the corpus word list. Acts as a cursor.

        corpus_name : str
            Name of the corpus ("Disco", "News") used for error messages.

        annotate_fn : Callable[[List[str]], str]
            Function producing the annotated coreference sentence.
            It receives the list of word IDs and must return a string.

        debug : bool, optional
            Enable debug-printing. Defaults to False.

        Returns
        -------
        Tuple[Optional[List[Any]], int]
            - structured match if found, otherwise None
            - updated word index

        Raises
        ------
        ValueError
            If a token mismatch occurs or if indexing goes out of range.
        """

        if sentence not in text_dict:
            return None, word_idx

        tokenized, alignments = text_dict[sentence]
        expected_tokens = tokenized.split()

        wids: List[str] = []

        for token in expected_tokens:
            current_wid, corpus_token = words_list[word_idx]

            # Allowed special case for quotes
            tokens_match = (
                corpus_token == token
                or (corpus_token == '``' and token == '"')
                or (corpus_token == "''" and token == '"')
            )

            if not tokens_match:
                raise ValueError(
                    f"Token mismatch in {corpus_name}: '{corpus_token}' vs '{token}' "
                    f"(previous: {words_list[word_idx-1][1]}, "
                    f"next: {words_list[word_idx+1][1]})"
                )

            wids.append(current_wid)
            word_idx += 1

        annotated = annotate_coref_sentence(wids, data=text_dict._parent_data)

        matched_struct = [
            sentence,       # 1. NMT input sentence
            tokenized,      # 2. Tokenized corpus sentence
            wids,           # 3. Unique token IDs
            alignments,     # 4. Word alignment data
            annotated       # 5. Annotated with coreference info
        ]

        if debug:
            print(f"[DEBUG] Matched: {sentence}")

        return matched_struct, word_idx

    # Prepare containers
    matched_data: List[List[Any]] = []

    disco_text = disco_src_pcf["text"]
    news_text = news_src_pcf["text"]

    # Attach parent references (used in annotate_coref_sentence)
    disco_text._parent_data = disco_src_pcf
    news_text._parent_data = news_src_pcf

    disco_words = list(disco_src_pcf["words"].items())
    news_words = list(news_src_pcf["words"].items())

    disco_idx = 0
    news_idx = 0

    # Main loop ---------------------------------------------------------------

    for sentence in src_nmt:
        if sentence in disco_text:
            struct, disco_idx = match_sentence_in_corpus(
                sentence, disco_text, disco_words, disco_idx, corpus_name="Disco"
            )
        elif sentence in news_text:
            struct, news_idx = match_sentence_in_corpus(
                sentence, news_text, news_words, news_idx, corpus_name="News"
            )
        else:
            raise ValueError(f"Sentence not found in any corpus: '{sentence}'")

        matched_data.append(struct)

        if verbose:
            print(f"Matched: {sentence}")

    return matched_data
