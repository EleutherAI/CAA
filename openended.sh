#!/bin/bash

python scoring.py

python plot_results.py $@ --layers 13 --multipliers -1.5 -1 0 1 1.5 --type open_ended  --title "Layer 13 - Llama 2 7B Chat"
python plot_results.py $@ --layers 14 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --title "Layer 14 - Llama 2 13B Chat"