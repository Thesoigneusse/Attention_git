# src/types.py

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union

# --- Type Aliases ---
# Représente une opération d'alignement : (Nom, Index Ref, Index Hyp)
# Exemple : ('match', 0, 0) ou ('ins', None, 1)
AlignmentOp = Tuple[str, Optional[int], Optional[int]]

@dataclass
class AppConfig:
    """
    Configuration globale de l'application.
    Contient tous les arguments CLI et les constantes de réglage.
    """
    # Chemins de fichiers (Paths)
    corpus_source: Path
    corpus_target: Path
    system_data: Path
    parcor_base_path: Path
    output_file: Path
    log_file: Optional[Path] = None 

    # Paramètres d'exécution
    evaluate_language: str = 'source'  # 'source' ou 'target'
    canmt_system: str = 'concat'       # 'concat' ou 'multienc'
    
    # Paramètres algorithmiques
    wer_threshold: float = 0.5
    coref_link_score_mode: str = 'max' # 'max' ou 'avg'
    
    # Options de débogage
    verbose: bool = False
    debug: bool = False


@dataclass
class Mention:
    """
    Représente une mention de coréférence extraite des fichiers XML ParCorFull.
    """
    id: str
    fileid: str
    span: str
    coref_class: str          # L'ID du cluster de coréférence (ex: "set_10")
    mention_text: str
    word_indices: List[int]   # Liste des IDs de mots (ex: [1, 2, 3])


@dataclass
class CorpusSentence:
    """
    Représente une phrase du corpus alignée et annotée.
    """
    raw_text: str             # Texte brut du fichier XML
    tokenized_text: str       # Texte après tokenization.py
    alignment: List[AlignmentOp] # Liste des opérations d'édition (Raw -> Tok)
    word_ids: List[str]       # IDs uniques des mots dans le corpus
    annotated_text: str       # Texte décoré (ex: "#[chat]#-set_1")


@dataclass
class CorpusData:
    """
    Conteneur pour tout le corpus chargé (DiscoMT ou News).
    """
    # Map: texte brut -> (texte tokenisé, alignement)
    text_map: Dict[str, Tuple[str, List[AlignmentOp]]] 
    
    # Map: 'prefix-word_id' -> 'Mot textuel'
    words: Dict[str, str]
    
    # Map: 'prefix-word_id' -> 'set_id' (seulement pour les mots dans une mention)
    words_in_coref: Dict[str, str]
    
    # Liste plate de toutes les mentions trouvées
    coref_mentions: List[Mention]


@dataclass
class SystemSequence:
    """
    Représente une entrée dans le fichier de sortie du système (Attention).
    """
    current: str                # La phrase générée/analysée
    context: str                # La phrase de contexte
    attention: List[List[float]] # Matrice d'attention [len(current) x len(context)]

    def to_pdf(path: Path) -> None:
        """
        Sauvegarde la matrice d'attention sous forme de fichier PDF.
        """
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        def visualize(pdTable, yy=0, xx=0):
            first_column = pdTable.columns[0]
            ll_y = len(pdTable)//4
            ll_y = (5 if ll_y < 5 else ll_y)
            ll_x = len(pdTable.columns)//3
            ll_x = (5 if ll_x < 5 else ll_x)
            yy = ll_y if yy == 0 else yy
            xx = ll_x if xx == 0 else xx
            fig, ax = plt.subplots(figsize=(xx,yy))
            # Set index AFTER heatmap to avoid issues with object dtype in index
            # pdTable = pdTable.set_index(first_column) # Remove this line
            numeric_data = pdTable.iloc[:, 1:].astype(float) # Ensure data is float

            mask = numeric_data == 0 # Apply mask to the numeric data
            sns.heatmap(numeric_data, vmin=0, vmax=1, cmap= sns.cm.rocket_r,
                            annot=True, mask=mask,
                            linecolor = 'gray', ax = ax,
                            linewidth=.5,)
            # Set yticklabels from the first column after the heatmap is drawn
            ax.set_yticklabels(pdTable[first_column])
            ax.set(xlabel="context", ylabel="current")
            ax.xaxis.tick_top()
            plt.xticks(rotation=70)
            plt.yticks(rotation=0)
            ax.xaxis.set_label_position('top')
            plt.tight_layout()
            # return fig

        


@dataclass
class AnalysisResult:
    """
    Résultat de l'évaluation d'une séquence.
    """
    seq_id: str
    current_sent: str
    context_sent: str
    
    # Liste des métriques pour chaque lien de coréférence trouvé.
    # Format d'un item : [IsAntecedent(bool), HasWeight(bool), Unused(bool), Score(float)]
    metrics: List[List[Union[bool, float]]] = field(default_factory=list)