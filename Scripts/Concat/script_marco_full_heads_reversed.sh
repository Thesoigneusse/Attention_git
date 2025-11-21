#!/bin/bash

# Define the base command
ATTENTION_GIT="$HOME/Attention_git"
base_command="python3 $ATTENTION_GIT/marco_script/match-nmt2parcorfull.py /home/getalp/lopezfab/Attention/k3/GOLD/test.en /home/getalp/lopezfab/Attention/k3/GOLD/test.de $ATTENTION_GIT/marco_script/list_filename_matrice/concat/full_matrice/list_chemin_0_0.txt --output-file=$ATTENTION_GIT/Output/concat_debug.results --canmt-system=concat --pudb=False --local=False"
# python3 ./marco_script/match-nmt2parcorfull.py /home/getalp/lopezfab/Attention/k3/GOLD/test.en /home/getalp/lopezfab/Attention/k3/GOLD/test.de ./marco_script/list_filename_matrice/concat/full_matrice/list_chemin_0_0.txt --output-file=./Output/concat_debug.results --canmt-system=concat --pudb=False
# Loop through the desired range
layer=0
    echo "layer: $layer"
    for token_level_heads in {7..0..-1}; do
        echo " * Token level head: $token_level_heads"
        # Modify the parameter
        modified_command=${base_command//list_chemin_0_0.txt/list_chemin_${layer}_${token_level_heads}.txt}
        # Execute the command and store the output in a file
        output_file="$ATTENTION_GIT/Output/Marco_script_output/output_${layer}_${token_level_heads}.txt"
        # output_file= python3 $ATTENTION_GIT/marco_script/match-nmt2parcorfull.py /home/getalp/lopezfab/Attention/k3/GOLD/test.en /home/getalp/lopezfab/Attention/k3/GOLD/test.de $ATTENTION_GIT/marco_script/list_filename_matrice/multi_enc/full_matrice/list_chemin_${sentence_level_head}_${token_level_heads}.txt --output-file=$ATTENTION_GIT/Output/debug.results --canmt-system=multienc --pudb=False
        $modified_command > $output_file
        echo "Output stored in $output_file"
    done

