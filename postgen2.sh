#!/bin/bash

if [ "$1" = "0" ]; then

    # python prompting_with_steering.py $2 --layers $(seq 0 31) --multipliers -1 0 1 --type ab 
    # python plot_results.py $2 --layers $(seq 0 31) --multipliers -1 1 --type ab 

    # python prompting_with_steering.py $2 --layers $(seq 0 35) --multipliers -1 0 1 --type ab --model_size "13b"
    # python plot_results.py $2 --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size "13b"

    python prompting_with_steering.py $2 --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector_model Llama-2-7b-hf
    python plot_results.py $2 --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector_model Llama-2-7b-hf --title "CAA transfer from base to chat model"

    python prompting_with_steering.py $2 --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector 13
    python plot_results.py $2 --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector 13 --title "CAA transfer from layer 13 vector to other layers"

elif [ "$1" = "1" ]; then

    # python prompting_with_steering.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab
    # python prompting_with_steering.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt pos
    # python prompting_with_steering.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt neg
    # python plot_results.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --title "Layer 13 - Llama 2 7B Chat"

    python prompting_with_steering.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b"
    python prompting_with_steering.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt pos
    python prompting_with_steering.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt neg
    python plot_results.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --title "Layer 14 - Llama 2 13B Chat"

elif [ "$1" = "2" ]; then

    # python prompting_with_steering.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type mmlu
    # python plot_results.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type mmlu

    # python prompting_with_steering.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy
    # python plot_results.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy

    python prompting_with_steering.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type mmlu --model_size "13b"
    python plot_results.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type mmlu --model_size "13b"

    python prompting_with_steering.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size "13b"
    python plot_results.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size "13b"

# elif [ "$1" = "3" ]; then

#     python prompting_with_steering.py $2 --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended
#     python prompting_with_steering.py $2 --layers 14 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --model_size "13b"

#     python scoring.py

#     python plot_results.py $2 --layers 13 --multipliers -1.5 -1 0 1 1.5 --type open_ended  --title "Layer 13 - Llama 2 7B Chat"
#     python plot_results.py $2 --layers 14 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --title "Layer 14 - Llama 2 13B Chat"

fi