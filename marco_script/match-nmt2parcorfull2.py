import os
import sys
import re

import torch
import argparse
import xml.etree.cElementTree as ET

import edit_distance
_VERBOSE = True
_CORPUS_SYSTEM_COREF_MATCHES = False
_CORPUS_SYSTEM_COMPARISON_LOG = False
_MENTION_LOG = False
_DEBUG_LOG = True
_FULL_MATRICE = True
_DEBUG_WER = False
_CTX_NEEDED_AND_HARD_COREF_CONCAT_IDS = False # Permet de se restreindre a un subset de test
_DATAPATH="./marco_script/data"
_TRAITEMENT="full_matrice"

parser = argparse.ArgumentParser(description='Performs alignment between corpus (ParCorFul2) data and system data and compute coreference resolution metrics over coreference links using attention weights as scores')
parser.add_argument('corpus_source', help='source language corpus data')
parser.add_argument('corpus_target', help='target language corpus data')
parser.add_argument('system_data', help='system data, either input or output (specified by --evaluate-language), to align to corpus data')
parser.add_argument('--evaluate-language', type=str, default='source', help='Specify which language is evaluated: source (default), target')
parser.add_argument('--canmt-system', type=str, default='concat', help='Specify which type of CA-NMT is evaluated: concat (default), multienc')
parser.add_argument('--output-file', type=str, default="./attention_analysis.results", help='Specify the path of the output file')
args = parser.parse_args()

wer_threshold = 0.5
coreference_link_score = 'max'  # 'max' or 'avg', but the script only apply use_avg_score = coreference_link_score == 'avg'
canmt_system = args.canmt_system   # 'multienc' or 'concat', but the script only apply cur_bogus_idx = 0 if canmt_system == 'concat' else 1
eval_language = args.evaluate_language    # 'source' or 'target'
output_file = args.output_file # output file

def read_txt(filename):

    f = open(filename, encoding='utf-8')
    ll = f.readlines()
    f.close()
    return [l.strip() for l in ll]

if __name__ == "__main__":
    src = read_txt(args.corpus_source)
    tgt = read_txt(args.corpus_target)
    print(src)
