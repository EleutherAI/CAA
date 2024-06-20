from gpu_graph import run
import sys

commands = {
    'gen_acts': 'python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --save_activations',
    'gen_orth': 'python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --method orth',
    'gen_quad': 'python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --method quad',
    'gen_logit': 'python generate_vectors.py --layers $(seq 0 31) --model_size "7b" --logit',
    'gen_acts13': 'python generate_vectors.py --layers $(seq 0 35) --model_size "13b" --save_activations',
    'gen_orth13': 'python generate_vectors.py --layers $(seq 0 35) --model_size "13b" --method orth',
    'gen_quad13': 'python generate_vectors.py --layers $(seq 0 35) --model_size "13b" --method quad',
    'gen_logit13': 'python generate_vectors.py --layers $(seq 0 35) --model_size "13b" --logit',

    'basic': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .5 3) --type ab --unnormalized',
    'layers': 'python prompting_with_steering.py --layers $(seq 0 31) --multipliers -1 0 1 --type ab --unnormalized',
    'leace': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .5 3) --type ab --unnormalized --leace',
    'orth': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .5 3) --type ab --unnormalized --leace --method orth',
    'quad': 'python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type ab --unnormalized --leace --method quad',
    'qall': 'python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type ab --unnormalized --leace --method qall',
    'logit': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .5 3) --type ab --unnormalized --logit',
    'class': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .5 3) --type ab --unnormalized --classify mean',
    'lda': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .5 3) --type ab --unnormalized --classify lda',
    'after': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .5 3) --type ab --unnormalized --after', 
}

dependencies = {job: [] for job in commands}

dependencies['basic'] = ['gen_acts']
dependencies['layers'] = ['gen_acts']
dependencies['leace'] = ['gen_acts']
dependencies['orth'] = ['gen_orth']
dependencies['quad'] = ['gen_quad']
dependencies['qall'] = ['gen_quad']
dependencies['logit'] = ['gen_logit']
dependencies['class'] = ['gen_acts']
dependencies['lda'] = ['gen_acts']
dependencies['after'] = ['gen_acts']


for job in commands:
    commands[job] += f"> logs/0620a_{job}.log 2>&1"

if __name__ == "__main__":
    # usage: python run_whatever.py 1,2,3,4,5,6,7
    if len(sys.argv) == 1:
        # default to all GPUs
        gpus = range(8)
    else:
        gpus = [int(gpu) for gpu in sys.argv[1].split(",")]

    for job in commands:
        print(f"{job}: {commands[job]} <-- {dependencies[job]}")
    print()
    print(f"Running on GPUs: {gpus}")

    run(gpus, commands, dependencies)