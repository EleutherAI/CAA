
#!/bin/bash

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab
python plot_results.py --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size "13b"

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --leace
python plot_results.py --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size "13b" --leace

python plot_results.py --layers $(seq 0 31) --multipliers -1 1 --type ab --leace --method orth
python plot_results.py --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size "13b" --leace --method orth
