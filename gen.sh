#!/bin/bash

python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" 
echo "GENERATED: 7b-chat"
python generate_vectors.py --layers $(seq 0 35) --model_size "13b"
echo "GENERATED: 13b-chat"
python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model 
echo "GENERATED: 7b"
python normalize_vectors.py
echo "NORMALIZED VECTORS"

python plot_activations.py --layers $(seq 0 31) --model_size "7b"
echo "PLOTTED ACTIVATIONS"
python analyze_vectors.py
echo "ANALYZED"