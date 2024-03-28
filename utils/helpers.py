import torch as t
import matplotlib.pyplot as plt
import einops as e

def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)



def steer_after_position(
        matrix, vector, position_ids, after=None, 
        eraser=None, threshold=None, class_vector=None,
        n_decomp=None):
    # n: sequence length
    # d: embedding size

    # matrix: (n, d)
    # vector: (d, )
    # position_ids: (n, )
    # after: int
    # eraser: function
    # threshold: float
    # class_vector: (d, )
    # n_decomp: int

    # mask: (n, )

    if after is None:
        mask = t.ones_like(position_ids, dtype=t.bool)
    else:
        mask = position_ids >= after

    if n_decomp is not None:
        # untested!
        
        h = matrix[mask]
        b, d = h.size(0), h.size(1)

        la = t.rand(b, n_decomp, device=h.device)
        a = t.softmax(la, dim=1)  # (b, n)
        eta = t.rand(b, n_decomp, device=h.device)
        a_norms = e.einsum(a, a, 'b n, b n -> b')
        a_pinv = a / a_norms[:, None]  # (b, n)
        a_eta = e.einsum(a, eta, 'b n, b n -> b')
        x = a_pinv * (1 - a_eta) + eta  # (b, n)
        G = e.einsum(x, h, 'b n, b d -> b n d')

        if threshold is not None:
            submask = G @ class_vector < threshold  # (b, n)
        else:
            submask = t.ones_like(x, dtype=t.bool)
        
        if eraser is not None:
            G[submask] = eraser(G[submask])
        if vector is not None:
            G[submask] += vector
        
        h = e.einsum(G, a, 'b n d, b n -> b d')
        matrix[mask] = h

        return matrix


    if threshold is not None:
        mask &= matrix @ class_vector < threshold
    
    if eraser is not None:
        matrix[mask] = eraser(matrix[mask])
    if vector is not None:
        matrix[mask] += vector

    return matrix


def find_last_subtensor_position(tensor, sub_tensor):
    n, m = tensor.size(0), sub_tensor.size(0)
    if m > n:
        return -1
    for i in range(n - m, -1, -1):
        if t.equal(tensor[i : i + m], sub_tensor):
            return i
    return -1


def find_instruction_end_postion(tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'


def get_model_path(size: str, is_base: bool):
    if is_base:
        return f"meta-llama/Llama-2-{size}-hf"
    else:
        return f"meta-llama/Llama-2-{size}-chat-hf"

def model_name_format(name: str) -> str:
    name = name.lower()
    is_chat = "chat" in name
    is_7b = "7b" in name
    if is_chat:
        if is_7b:
            return "Llama 2 Chat 7B"
        else:
            return "Llama 2 Chat 13B"
    else:
        if is_7b:
            return "Llama 2 7B"
        else:
            return "Llama 2 13B"
