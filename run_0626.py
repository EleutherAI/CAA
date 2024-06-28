from gpu_graph import run
import sys

name = "0626"

commands = {
    'gen_acts': 'python generate_vectors.py --layers $(seq 0 31) --save_activations',
    'gen_orth': 'python generate_vectors.py --layers $(seq 0 31) --method orth',
    'gen_quad': 'python generate_vectors.py --layers $(seq 0 31) --method quad',
    'gen_logit': 'python generate_vectors.py --layers $(seq 0 31) --logit',

    'allcaa': f'python prompting_with_steering.py --layers -1 --multipliers $(seq -.5 .025 .5) --type ab --unnormalized',
    'allleace': f'python prompting_with_steering.py --layers -1 --multipliers $(seq -3 .25 3) --type ab --unnormalized --leace',
    'allorth': f'python prompting_with_steering.py --layers -1 --multipliers $(seq -3 .25 3) --type ab --unnormalized --leace --method orth',
    'basic': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized',
    'layers': 'python prompting_with_steering.py --layers $(seq 10 20) --multipliers $(seq -3 .25 3) --type ab --unnormalized',
    'leace': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized --leace',
    'orth': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized --leace --method orth',
    'quad': 'python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type ab --unnormalized --leace --method quad',
    'qall': 'python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type ab --unnormalized --leace --method qall',
    'logit': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized --logit',
    'class': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized --classify mean',
    'lda': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized --classify lda',
    'quadlda': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized --classify lda --leace --method quad',
    'alltok': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -3 .25 3) --type ab --unnormalized --all_tokens', 
}

for job in list(commands.keys()):

    if "quad" in job or "qall" in job:
        continue

    cmd = commands[job]
    cmd += ' --model_size "13b"'
    cmd = cmd.replace("layers 13", "layers 14")
    cmd = cmd.replace("seq 0 31", "seq 0 39")

    commands[job + '_13'] = cmd

for job in list(commands.keys()):

    cmd = commands[job]
    cmd += ' --open'
    cmd = cmd.replace("seq -3 .25 3", "seq -.15 .01 .15")
    cmd = cmd.replace("seq -.5 .025 .5", "seq -.05 .0025 .05")
    commands[job + "_open"] = cmd

dependencies = {}

for job in commands:
    if job.startswith("gen_"):
        dependencies[job] = []
        continue

    base = job.split("_")[0]

    dep_base = {
        "allcaa": "gen_acts",
        "allleace": "gen_acts",
        "allorth": "gen_orth",
        "basic": "gen_acts",
        "layers": "gen_acts",
        "leace": "gen_acts",
        "orth": "gen_orth",
        "quad": "gen_quad",
        "qall": "gen_quad",
        "logit": "gen_logit",
        "class": "gen_acts",
        "lda": "gen_acts",
        "quadlda": "gen_quad",
        "alltok": "gen_acts",
    }

    dep = dep_base[base]

    if job.endswith("_13"):
        dep += "_13"
    
    if job.endswith("_open"):
        dep += "_open"

    dependencies[job] = [dep]


for job in commands:
    commands[job] += f" > logs/{name}_{job}.log 2>&1"

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