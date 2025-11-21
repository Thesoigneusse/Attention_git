#!/bin/bash

# Define the base command
ATTENTION_GIT="$HOME/Attention_git"
SCRIPT="$ATTENTION_GIT/marco_script/match-nmt2parcorfull.py"
GOLD_EN="$HOME/Attention/k3/GOLD/test.en"
GOLD_DE="$HOME/Attention/k3/GOLD/test.de"
LISTE_CHEMIN_MATRICE="$ATTENTION_GIT/marco_script/list_filename_matrice/multi_enc/full_matrice/list_chemin_0_0.txt"
OUTPUT_FILE="$ATTENTION_GIT/Output/debug.results_2"
CANMT_SYSTEM="multienc"

base_command="python3 $SCRIPT $GOLD_EN $GOLD_DE $LISTE_CHEMIN_MATRICE --output-file=$OUTPUT_FILE --canmt-system=$CANMT_SYSTEM --pudb=False"

# Loop through the desired range
sentence_level_head=0
echo "Sentence level head: $sentence_level_head"
token_level_heads=7
echo " * Token level head: $token_level_heads"
# Modify the parameter
modified_command=${base_command//list_chemin_0_0.txt/list_chemin_${sentence_level_head}_${token_level_heads}.txt}
# Execute the command and store the output in a file
output_file="$ATTENTION_GIT/Output/Marco_script_output/multi_enc/V2/output_${sentence_level_head}_${token_level_heads}.txt"
# output_file= python3 $ATTENTION_GIT/marco_script/match-nmt2parcorfull.py /home/getalp/lopezfab/Attention/k3/GOLD/test.en /home/getalp/lopezfab/Attention/k3/GOLD/test.de $ATTENTION_GIT/marco_script/list_filename_matrice/multi_enc/full_matrice/list_chemin_${sentence_level_head}_${token_level_heads}.txt --output-file=$ATTENTION_GIT/Output/debug.results --canmt-system=multienc --pudb=False
$modified_command > $output_file
echo "Output stored in $output_file"
