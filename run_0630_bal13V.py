from gpu_graph import run
import sys

name = "0630_bal13V"

commands = {
    # 'gen_basic': 'python generate_vectors.py --layers $(seq 0 31)',
    # 'gen_acts': 'python generate_vectors.py --layers $(seq 0 31) --save_activations',
    # 'gen_orth': 'python generate_vectors.py --layers $(seq 0 31) --method orth',
    # 'gen_quad': 'python generate_vectors.py --layers $(seq 0 31) --method quad',
    # 'gen_logit': 'python generate_vectors.py --layers $(seq 0 31) --logit',

    'allcaa': f'python prompting_with_steering.py --layers -1 --multipliers $(seq -.25 .1 .25) --type open_ended --unnormalized',
    'allleace': f'python prompting_with_steering.py --layers -1 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --leace',
    'allorth': f'python prompting_with_steering.py --layers -1 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --leace --method orth',
    'basic': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized',
    # 'layers': 'python prompting_with_steering.py --layers $(seq 10 20) --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized',
    'leace': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --leace',
    'orth': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --leace --method orth',
    'quad': 'python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type open_ended --unnormalized --leace --method quad',
    'qall': 'python prompting_with_steering.py --layers 13 --multipliers -1 0 1 --type open_ended --unnormalized --leace --method qall',
    'logit': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --logit',
    'class': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --classify mean',
    'lda': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --classify lda',
    'quadlda': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --classify lda --leace --method quad',
    'alltok': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --all_tokens', 
    'alltokleace': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --all_tokens --leace', 
    'alltokorth': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --all_tokens --leace --method orth', 
    'instr': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --only_instr', 
    'instrleace': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --only_instr --leace', 
    'instrorth': 'python prompting_with_steering.py --layers 13 --multipliers $(seq -2.5 2.5) --type open_ended --unnormalized --only_instr --leace --method orth', 
}

done = ['gen_basic', 'gen_orth', 'gen_quad', 'gen_logit']

for job in list(commands.keys()):
    #if not job.startswith("gen_"):
    commands[job] += ' --balanced'

for job in list(commands.keys()):

    if "quad" in job or "qall" in job:
        continue

    cmd = commands[job]
    cmd += ' --model_size "13b"'
    cmd = cmd.replace("layers 13", "layers 14")
    cmd = cmd.replace("seq 0 31", "seq 0 39")

    commands[job + '_13'] = cmd

commands = {k: v for k, v in commands.items() if "_13" in k}

# for job in list(commands.keys()):
#     cmd = commands[job]
#     cmd = cmd.replace("--type open_ended", "--type open_ended")

#     commands[job + '_openended'] = cmd

# for job in list(commands.keys()):
    
#     if "acts" in job or "logit" in job:
#         continue

#     cmd = commands[job]
#     cmd += ' --open'

#     commands[job + "_open"] = cmd

dependencies = {job: [] for job in commands}


dep_base = {
    "allcaa": ["gen_basic"],
    "allleace": ["gen_basic"],
    "allorth": ["gen_orth"],
    "basic": ["gen_basic"],
    "layers": ["gen_basic"],
    "leace": ["gen_basic"],
    "orth": ["gen_orth"],
    "quad": ["gen_quad"],
    "qall": ["gen_quad"],
    "logit": ["gen_logit"],
    "class": ["gen_basic"],
    "lda": ["gen_basic"],
    "quadlda": ["gen_quad", "gen_basic"],
    "alltok": ["gen_basic"],
    "alltokleace": ["gen_basic"],
    "alltokorth": ["gen_orth"],
    "instr": ["gen_basic"],
    "instrleace": ["gen_basic"],
    "instrorth": ["gen_orth"],
}

for k in dep_base:
    for d in dep_base[k]:
        if d in done:
            dep_base[k].remove(d)

for job in commands:
    if job.startswith("gen_"):
        dependencies[job] = []
        continue

    base = job.split("_")[0]

    deps = dep_base[base]

    if job.endswith("_13"):
        deps = [dep + "_13" for dep in deps]
    
    if job.endswith("_open"):
        deps = [dep + "_open" for dep in deps]

    dependencies[job] = deps


for job in commands:
    commands[job] += f" > logs/{name}_{job}.log 2>&1"

if __name__ == "__main__":
    # usage: python run_whatever.py 1,2,3,4,5,6,7
    if len(sys.argv) == 1:
        # default to all GPUs
        gpus = list(range(8))
    else:
        gpus = [int(gpu) for gpu in sys.argv[1].split(",")]

    for job in commands:
        print(f"{job}: {commands[job]} <-- {dependencies[job]}")
    print()
    print(f"Running on GPUs: {gpus}")

    run(gpus, commands, dependencies)