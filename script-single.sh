#!/bin/bash

if [ "$1" = "7b-chat" ]; then
    if [ "$2" != "--leace" ]; then
        echo "GENERATING..."
        python generate_vectors.py --layers $(seq 0 31) --save_activations --model_size "7b" 
        echo "GENERATED: 7b-chat"
        python normalize_vectors.py
        echo "NORMALIZED VECTORS"
        python plot_activations.py --layers $(seq 0 31) --model_size "7b"
        echo "PLOTTED ACTIVATIONS"
        python analyze_vectors.py
        echo "ANALYZED"
    fi
    echo "PROMPTING..."
    python prompting_with_steering.py $2 --layers $(seq 0 31) --multipliers -1 0 1 --type ab 
    python prompting_with_steering.py $2 --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector 13
    python prompting_with_steering.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab
    python prompting_with_steering.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt pos
    python prompting_with_steering.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --system_prompt neg
    python prompting_with_steering.py $2 --layers 13 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended
    python prompting_with_steering.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type mmlu
    python prompting_with_steering.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy
    echo "PROMPTED"
    python plot_results.py $2 --layers 13 --multipliers -1 -0.5 0 0.5 1 --type ab --title "Layer 13 - Llama 2 7B Chat"
    python plot_results.py $2 --layers $(seq 0 31) --multipliers -1 1 --type ab 
    python plot_results.py $2 --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector 13 --title "CAA transfer from layer 13 vector to other layers"
    python plot_results.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type mmlu
    python plot_results.py $2 --layers 13 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy
    echo "SCORING..."
    python scoring.py
    echo "SCORED"
    python plot_results.py $2 --layers 13 --multipliers -1.5 -1 0 1 1.5 --type open_ended  --title "Layer 13 - Llama 2 7B Chat"
    echo "PLOTTED"
elif [ "$1" = "13b-chat" ]; then
    if [ "$2" != "--leace" ]; then
        echo "GENERATING..."
        python generate_vectors.py --layers $(seq 0 35) --model_size "13b"
        echo "GENERATED: 13b-chat"
        python normalize_vectors.py
        echo "NORMALIZED VECTORS"
        python analyze_vectors.py
        echo "ANALYZED"
    fi
    echo "PROMPTING..."

    python prompting_with_steering.py $2 --layers $(seq 0 35) --multipliers -1 0 1 --type ab --model_size "13b"

    python prompting_with_steering.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b"

    python prompting_with_steering.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt pos

    python prompting_with_steering.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --system_prompt neg

    python prompting_with_steering.py $2 --layers 14 --multipliers -2.0 -1.5 -1 0 1 1.5 2.0 --type open_ended --model_size "13b"

    python prompting_with_steering.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type mmlu --model_size "13b"

    python prompting_with_steering.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size "13b"
    echo "PROMPTED"
    python plot_results.py $2 --layers $(seq 0 35) --multipliers -1 1 --type ab --model_size "13b"

    python plot_results.py $2 --layers 14 --multipliers -1 -0.5 0 0.5 1 --type ab --model_size "13b" --title "Layer 14 - Llama 2 13B Chat"

    python plot_results.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type mmlu --model_size "13b"

    python plot_results.py $2 --layers 14 --multipliers -2 -1 0 1 2 --type truthful_qa --behaviors sycophancy --model_size "13b"
    echo "SCORING..."
    python scoring.py
    echo "SCORED"
    python plot_results.py $2 --layers 14 --multipliers -1.5 -1 0 1 1.5 --type open_ended --model_size "13b" --title "Layer 14 - Llama 2 13B Chat"
    echo "PLOTTED"
elif [ "$1" = "7b-transfer" ]; then
    if [ "$2" != "--leace" ]; then
        echo "GENERATING..."

        python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --use_base_model 
        echo "GENERATED: 7b"
        python normalize_vectors.py
        echo "NORMALIZED VECTORS"

        python analyze_vectors.py
        echo "ANALYZED"
    fi
    echo "PROMPTING..."
    python prompting_with_steering.py $2 --layers $(seq 0 31) --multipliers -1 0 1 --type ab --override_vector_model Llama-2-7b-hf
    echo "PROMPTED"
    python plot_results.py $2 --layers $(seq 0 31) --multipliers -1 1 --type ab --override_vector_model Llama-2-7b-hf --title "CAA transfer from base to chat model"
    echo "PLOTTED"
else
    echo "what?"
fi