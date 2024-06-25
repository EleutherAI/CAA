#!/bin/bash

task=$1
shift

if [[ "$task" == *"0"* ]]; then

    python prompting_with_steering.py $@ --layers $(seq 0 31) --multipliers -2 0 2 --type ab 
    python plot_results.py $@ --layers $(seq 0 31) --multipliers -2 2 --type ab 

    python prompting_with_steering.py $@ --layers $(seq 0 35) --multipliers -2 0 2 --type ab --model_size "13b"
    python plot_results.py $@ --layers $(seq 0 35) --multipliers -2 2 --type ab --model_size "13b"

    python prompting_with_steering.py $@ --layers $(seq 0 31) --multipliers -2 0 2 --type ab --override_vector_model Llama-2-7b-hf
    python plot_results.py $@ --layers $(seq 0 31) --multipliers -2 2 --type ab --override_vector_model Llama-2-7b-hf --title "CAA transfer from base to chat model"

    python prompting_with_steering.py $@ --layers $(seq 0 31) --multipliers -2 0 2 --type ab --override_vector 13
    python plot_results.py $@ --layers $(seq 0 31) --multipliers -2 2 --type ab --override_vector 13 --title "CAA transfer from layer 13 vector to other layers"
fi
if [[ "$task" == *"1"* ]]; then

    python prompting_with_steering.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type ab
    python prompting_with_steering.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type ab --system_prompt pos
    python prompting_with_steering.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type ab --system_prompt neg
    python plot_results.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type ab --title "Layer 13 - Llama 2 7B Chat"

    python prompting_with_steering.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type ab --model_size "13b"
    python prompting_with_steering.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type ab --model_size "13b" --system_prompt pos
    python prompting_with_steering.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type ab --model_size "13b" --system_prompt neg
    python plot_results.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type ab --model_size "13b" --title "Layer 14 - Llama 2 13B Chat"
fi
if [[ "$task" == *"2"* ]]; then

    python prompting_with_steering.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type mmlu
    python plot_results.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type mmlu

    python prompting_with_steering.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type truthful_qa --behaviors sycophancy
    python plot_results.py $@ --layers 13 --multipliers -4 -2 0 2 4 --type truthful_qa --behaviors sycophancy

    python prompting_with_steering.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type mmlu --model_size "13b"
    python plot_results.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type mmlu --model_size "13b"

    python prompting_with_steering.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type truthful_qa --behaviors sycophancy --model_size "13b"
    python plot_results.py $@ --layers 14 --multipliers -4 -2 0 2 4 --type truthful_qa --behaviors sycophancy --model_size "13b"
fi
if [[ "$task" == *"3"* ]]; then

    python prompting_with_steering.py $@ --layers 13 --multipliers -4 -3 -2 0 2 3 4 --type open_ended
    python prompting_with_steering.py $@ --layers 14 --multipliers -4 -3 -2 0 2 3 4 --type open_ended --model_size "13b"

    python scoring.py

    python plot_results.py $@ --layers 13 --multipliers -3 -2 0 2 3 --type open_ended  --title "Layer 13 - Llama 2 7B Chat"
    python plot_results.py $@ --layers 14 --multipliers -3 -2 0 2 3 --type open_ended --model_size "13b" --title "Layer 14 - Llama 2 13B Chat"

fi