"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --model_size 7b --behaviors sycophancy
"""

import json
from concept_erasure import LeaceFitter, QuadraticFitter
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_open_response_data_path,
    get_vector_path,
    get_eraser_path,
    get_activations_path,
    ALL_BEHAVIORS,
    force_save,
)

load_dotenv()

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")


class ComparisonDataset(Dataset):
    def __init__(self, data_path, token, model_name_path, use_chat):
        with open(data_path, "r") as f:
            self.data = json.load(f)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path, token=token
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat

    def prompt_to_tokens(self, instruction, model_output):
        if self.use_chat:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        else:
            tokens = tokenize_llama_base(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return p_tokens, n_tokens

def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: List[str],
    model: LlamaWrapper,
    leace_method: str,
    logit: bool,
    stdev: bool,
    open_response: bool,
):
    if open_response:
        data_path = get_open_response_data_path(behavior)
    else:
        data_path = get_ab_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = {layer: [] for layer in layers}
    if logit:
        AB_logits = {layer: [] for layer in layers}
    else:
        neg_activations = {layer: [] for layer in layers}

    dataset = ComparisonDataset(
        data_path,
        HUGGINGFACE_TOKEN,
        model.model_name_path,
        model.use_chat,
    )

    fitters : dict[int, LeaceFitter] = {l: None for l in layers}

    for layer in layers:
        act_dim = model.model.model.layers[layer].block.hidden_size
        if leace_method == "quad":
            assert not logit, "can't do logit with quadratic fitter"
            fitters[layer] = QuadraticFitter(
                act_dim, 3,
                device=model.device,
                dtype=t.float64,
            )
        else:
            fitters[layer] = LeaceFitter(
                act_dim, (1 if logit else 2),
                device=model.device,
                method=leace_method,
                dtype=t.float64,
            )

    for p_tokens, n_tokens in tqdm(dataset, desc="Processing prompts"):
        p_tokens = p_tokens.to(model.device)
        n_tokens = n_tokens.to(model.device)
        assert p_tokens.shape[0] == n_tokens.shape[0] == 1, f"data too long: {p_tokens.shape} {n_tokens.shape}"
        model.reset_all()

        p_tok = p_tokens[0, -2]
        n_tok = n_tokens[0, -2]
        logits = model.get_logits(p_tokens)[0, -2, (n_tok, p_tok)].detach()

        for layer in layers:
            # read *output* position [A/B] for logit mode
            # *input* position for non-logit mode
            # *last* position for open response
            if logit:
                position = -3
            elif not open_response:
                position = -2
            else:
                position = -1

            p_activations = model.get_last_activations(layer)[0, position, :].detach()
            pos_activations[layer].append(p_activations.cpu())

            x = p_activations[None]

            if logit:
                AB_logit = (logits[1] - logits[0])
                AB_logits[layer].append(AB_logit.cpu())

                z = AB_logit[None][None]
                fitters[layer].update(x, z)
            elif leace_method == "quad":
                fitters[layer].update_single(x, 1)
                fitters[layer].update_single(x, 2)
            else:
                z = t.tensor([[0, 1]], device=x.device)
                fitters[layer].update(x, z)
        if not logit:
            model.reset_all()
            model.get_logits(n_tokens)
            for layer in layers:
                n_activations = model.get_last_activations(layer)[0, -2, :].detach()
                neg_activations[layer].append(n_activations.cpu())
            
                x = n_activations[None]

                if leace_method == "quad":
                    fitters[layer].update_single(x, 0)
                    fitters[layer].update_single(x, 2)
                else:
                    z = t.tensor([[1, 0]], device=x.device)
                    fitters[layer].update(x, z)

    for layer in tqdm(layers, desc="Saving artifacts"):
        all_pos_layer = t.stack(pos_activations[layer])
        if logit:
            all_AB_logits = t.stack(AB_logits[layer])
            Exz = t.einsum("i,ik->ik", all_AB_logits, all_pos_layer).mean(dim=0)
            ExEz = all_AB_logits.mean() * all_pos_layer.mean(dim=0)
            vec = Exz - ExEz
        else:
            all_neg_layer = t.stack(neg_activations[layer])
            vec = (all_pos_layer - all_neg_layer).mean(dim=0)
            mean = (all_pos_layer + all_neg_layer).mean(dim=0) / 2
        eraser = fitters[layer].editor() if leace_method == "quad" else fitters[layer].eraser

        if leace_method == "leace":
            sigma = fitters[layer].sigma_xx.to(vec.device)
            L = t.linalg.cholesky(sigma + 1e-6 * t.eye(sigma.shape[0], device=sigma.device))
            precision = t.cholesky_inverse(L)
            
            vec64 = vec.to(precision.dtype)
            
            if stdev:
                vec /= (vec64 @ precision @ vec64).sqrt().to(vec.dtype)

            lda = (precision @ vec64).to(vec.dtype)

            force_save(
                lda,
                get_vector_path(behavior, layer, model.model_name_path, logit, stdev, open_response=open_response, prefix="lda"),
            )

        force_save(
            vec,
            get_vector_path(behavior, layer, model.model_name_path, logit, stdev, open_response=open_response),
        )
        if not logit:
            force_save(
                mean,
                get_vector_path(behavior, layer, model.model_name_path, logit, stdev, open_response=open_response, prefix="mean"),
            )
        force_save(
            eraser,
            get_eraser_path(behavior, layer, model.model_name_path, logit, open_response, leace_method),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )

def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
    **kwargs
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    behaviors: behaviors to generate vectors for
    leace_method: method to use for LEACE, either "leace" or "orth"
    logit: if True, weight by pos/neg logit instead of subtracting means
    """
    model = LlamaWrapper(
        HUGGINGFACE_TOKEN, size=model_size, use_chat=not use_base_model
    )
    for behavior in behaviors:
        print(f"Generating vectors for {behavior}")
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model, **kwargs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)
    parser.add_argument("--method", type=str, choices=["leace", "orth", "quad"], default="leace")
    parser.add_argument("--logit", action="store_true", default=False)
    parser.add_argument("--stdev", action="store_true", default=False)
    parser.add_argument("--open", action="store_true", default=False)

    args = parser.parse_args()
    if args.method != "leace" and args.stdev:
        raise ValueError("Can't use stdev with method other than leace")
    if args.open and args.logit:
        raise ValueError("Can't use logit with open response")
    if args.save_activations and args.open:
        raise NotImplementedError("Can't save activations with open response")
    if args.save_activations and args.logit:
        raise NotImplementedError("Can't save activations with logit")

    

    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors,
        leace_method=args.method,
        logit=args.logit,
        stdev=args.stdev,
        open_response=args.open,
    )

    print("Done!")