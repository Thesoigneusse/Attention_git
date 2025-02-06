#!/bin/bash

# Define the base command
ATTENTION_GIT="$HOME/Attention_git"
base_command="python3 $ATTENTION_GIT/marco_script/match-nmt2parcorfull.py /home/getalp/lopezfab/Attention/k3/GOLD/test.en /home/getalp/lopezfab/Attention/k3/GOLD/test.de $ATTENTION_GIT/marco_script/list_filename_matrice/multi_enc/full_matrice/list_chemin_0_0.txt --output-file=$ATTENTION_GIT/Output/debug.results --canmt-system=multienc --pudb=False"

# Loop through the desired range
for sentence_level_head in {0..7}; do
    for token_level_heads in {0..7}; do
        # Modify the parameter
        modified_command=${base_command//list_chemin_0_0_tsv.txt/list_chemin_${sentence_level_head}_${token_level_heads}_tsv.txt}
        # Execute the command and store the output in a file
        output_file="$ATTENTION_GIT/Output/Marco_script_output/output_$sentence_level_head_$token_level_heads.txt"
        $modified_command >> $output_file
    done
done