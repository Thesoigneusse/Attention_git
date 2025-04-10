# data_loader.py
import json
import xml.etree.cElementTree as ET
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class CorpusData:
    text: Dict[str, Tuple[str, List[str]]]
    words: Dict[str, str]
    coref: List[Dict]
    words_in_coref: Dict[str, str]

def load_corpus_data(file_path: str) -> CorpusData:
    """Charge les données du corpus à partir d'un fichier XML."""
    # Implémentation de la lecture des fichiers XML et conversion en CorpusData
    file_type = file_path.split('.')[-1]
    if file_type == 'xml':
        return load_corpus_data_xml(file_path)
    elif file_type == 'json':
        return load_corpus_data_json(file_path)
    else:
        raise ValueError("Format de fichier non supporté")

def load_system_data(file_path: str) -> Dict[str, str]:
    """Charge les données du système à partir d'un fichier texte."""
    # Implémentation de la lecture des fichiers texte et conversion en dictionnaire
    pass

def load_corpus_data_xml(file_path: str) -> CorpusData:
    """Charge les données du corpus à partir d'un fichier XML."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    text = {}
    words = {}
    coref = []
    words_in_coref = {}
    for child in root:
        if child.tag == 'TEXT':
            text[child.attrib['ID']] = (child.text, child.attrib['TYPE'])
        elif child.tag == 'WORDS':
            for word in child:
                words[word.attrib['ID']] = word.text
        elif child.tag == 'COREF':
            for coref_group in child:
                coref_group_list = []
                for word in coref_group:
                    coref_group_list.append(word.attrib['ID'])
                    words_in_coref[word.attrib['ID']] = word.text
                coref.append(coref_group_list)
    return CorpusData(text, words, coref, words_in_coref)

def load_corpus_data_json(file_path: str) -> CorpusData:
    """Charge les données du corpus à partir d'un fichier JSON."""
    raise NotImplementedError

if __name__ == '__main__':
    corpus_data = load_corpus_data('data.xml')
    print(corpus_data)