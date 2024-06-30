import torch as t

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from typing import Dict, Any, List
import json

import einops as e


import os, sys
dir2 = os.path.abspath('')
dir1 = dir2  # os.path.dirname(dir2)
if not dir1 in sys.path: sys.path.append(dir1)

from steering_settings import SteeringSettings
from behaviors import (
    get_open_ended_test_data,
    get_steering_eraser,
    change_eraser_dtype,
    get_steering_vector,
    get_system_prompt,
    get_truthful_qa_data,
    get_mmlu_data,
    get_ab_test_data,
    ALL_BEHAVIORS,
    get_results_dir,
)

from concept_erasure import LeaceFitter


BASE_DIR = dir1

def get_results_dir2(settings: SteeringSettings) -> str:
    return os.path.join(BASE_DIR, "results", settings.behavior)
    
    # elif settings.leace:
    #     return os.path.join(BASE_DIR, "output/0302/results", settings.behavior)
    # else:
    #     return os.path.join(BASE_DIR, "output/0229/results", settings.behavior)

def get_data(
    layer: int,
    multiplier: float,
    settings: SteeringSettings,
) -> Dict[str, Any]:
    directory = get_results_dir2(settings)
    settings.normalized = settings.normalized and not settings.stdev
    if settings.type == "open_ended":
        directory = directory.replace("results", os.path.join("results", "open_ended_scores"))
    # print(f"getting data from {directory}")
    filenames = settings.filter_result_files_by_suffix(
        directory, layer=layer, multiplier=multiplier
    )
    if len(filenames) > 1:
        print(f"[WARN] >1 filename found for layer {layer} mult {multiplier} and filter {settings}", filenames)
    if len(filenames) == 0:
        print(f"[WARN] no filenames found for layer {layer} mult {multiplier} and filter {settings}")
        return []
    # print(f"opening {filenames[0]}", flush=True)
    with open(filenames[0], "r") as f:
        # print(f"loading {filenames[0]}", flush=True)
        return json.load(f)

def get_mults(
    layer: int,
    settings: SteeringSettings,
) -> List[float]:
    directory = get_results_dir2(settings)
    settings.normalized = settings.normalized and not settings.stdev
    if settings.type == "open_ended":
        directory = directory.replace("results", os.path.join("results", "open_ended_scores"))
    filenames = settings.filter_result_files_by_suffix(
        directory, layer=layer
    )
    return list(sorted([float(f.split("multiplier=")[1].split("__")[0]) for f in filenames]))


def get_probs(
    layer: int,
    multiplier: float,
    settings: SteeringSettings,
    matching: bool,
) -> List[float]:
    data = get_data(layer, multiplier, settings)
    # print('got data', flush=True)
    match_probs, unmatch_probs = [], []
    for d in data:
        if matching:
            mat = d["answer_matching_behavior"][-2].lower()
            unmat = d["answer_not_matching_behavior"][-2].lower()
        else:
            mat, unmat = "a", "b"
        if mat not in 'ab' or unmat not in 'ab':
            # print(f"[WARN] match or unmatch not in 'ab': {mat}, {unmat}; {settings.behavior}")
            continue
        match_probs.append(d[f"{mat}_prob"])
        unmatch_probs.append(d[f"{unmat}_prob"])

    return np.array(match_probs), np.array(unmatch_probs)

def get_mprobs(
    layer: int,
    multiplier: float,
    settings: SteeringSettings,
    matching: bool = True,
) -> List[float]:
    match_probs, unmatch_probs = get_probs(layer, multiplier, settings, matching)
    return match_probs / (match_probs + unmatch_probs)

def get_acts_dir(settings: SteeringSettings) -> str:
    return os.path.join(BASE_DIR, "activations", settings.behavior)

def get_acts(settings: SteeringSettings, layer: int):
    print(f"getting activations for {settings.behavior} layer {layer}")
    directory = get_acts_dir(settings)
    for sign in ["pos", "neg"]:
        filename = os.path.join(directory, f"activations_{sign}_{layer}_Llama-2-7b-chat-hf.pt")
        with open(filename, "rb") as f:
            yield t.load(f)

def get_vec(behavior, stdev=False, open_response=False, layer=12, prefix="vec"):
    vec_path = os.path.join(BASE_DIR, "vectors", behavior,
        "open" if open_response else "",
        "stdev" if stdev else "", 
        f"{prefix}_layer_{layer}_Llama-2-7b-chat-hf.pt")
    print("vec_path", vec_path)
    with open(vec_path, "rb") as f:
        vec = t.load(f)

    nvec_path = os.path.join(BASE_DIR, "normalized_vectors", behavior,
        "open" if open_response else "",
        "stdev" if stdev else "", 
        f"vec_layer_{layer}_Llama-2-7b-chat-hf.pt")
    
    # check if file exists
    if not os.path.exists(nvec_path):
        print(f"File {nvec_path} does not exist")
        nvec = None
    else:
        print("nvec_path", nvec_path)
        with open(nvec_path, "rb") as f:
            nvec = t.load(f)

    return vec, nvec

def fit_lda(pos, neg, vec):
    act_dim = pos.shape[1]
    fitter = LeaceFitter(
                    act_dim, (2),
                    method="leace",
                    dtype=t.float64,
                )

    n = pos.shape[0]

    z_pos = t.zeros((n, 2), dtype=t.float64)
    z_pos[:, 1] = 1
    fitter.update(pos, z_pos)

    z_neg = t.zeros((n, 2), dtype=t.float64)
    z_neg[:, 0] = 1
    fitter.update(neg, z_neg)

    sigma = fitter.sigma_xx

    L = t.linalg.cholesky(sigma) # + 1e-6 * t.eye(sigma.shape[0], device=sigma.device))
    precision = t.cholesky_inverse(L)
    vec64 = vec.to(precision.dtype)

    lda = (precision @ vec64).to(vec.dtype)

    return lda, precision, fitter

CMAP = "viridis"

def hist2d_grid(probs_pairs, title="7B-chat", axis_labels=("x", "y")):
    # grid of 2D histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)

    for i, behavior in enumerate(ALL_BEHAVIORS):
        ax = axs[i//3, i%3]
        ax.set_facecolor('black')
        ax.set_aspect('equal')
        mprobs, mprobs_leace = probs_pairs[behavior]
        ax.hist2d(mprobs, mprobs_leace, bins=50, cmap=CMAP, cmin=1,
                    range=[[0, 1], [0, 1]])
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        ax.set_title(f"{behavior}: matching probs")
    
    plt.show()

def hist1d_grid(probs_pairs, title="7B-chat", axis_labels=("x", "y")):
    # grid of 1D paired histograms
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(title)

    for i, behavior in enumerate(ALL_BEHAVIORS):
        ax = axs[i//3, i%3]
        mprobs, mprobs_leace = probs_pairs[behavior]
        ax.hist(mprobs, bins=50, color="blue", alpha=0.5, label=axis_labels[0])
        ax.hist(mprobs_leace, bins=50, color="red", alpha=0.5, label=axis_labels[1])
        ax.legend()
        ax.set_title(f"{behavior}: matching probs")
    
    plt.show()

def make_grids(layer: int, multiplierx: float, multipliery=None, 
        settings_x: SteeringSettings=None,
        settings_y: SteeringSettings=None,
        axis_labels=("x", "y"),
        plot_title="Untitled Plot",
        ):
    probs_pairs = {}
    if multipliery is None:
        multipliery = multiplierx
    for behavior in ALL_BEHAVIORS:
        settings_x.behavior = behavior
        settings_y.behavior = behavior
        mprobs_x = get_mprobs(layer, multiplierx, settings_x) # x
        mprobs_y = get_mprobs(layer, multipliery, settings_y) # y

        probs_pairs[behavior] = (mprobs_x, mprobs_y)
    
    hist1d_grid(probs_pairs, title=plot_title, axis_labels=axis_labels)
    hist2d_grid(probs_pairs, title=plot_title, axis_labels=axis_labels)

def mults_plot(layers: list[int], multipliers: list[float], 
    settingses: dict[str, (SteeringSettings, str)],
    behaviors: list[str] = ALL_BEHAVIORS,
    rescale=False,
    ):
    if multipliers is None:
        mult_lists = [get_mults(layer, pair[0]) for layer, pair in zip(layers, settingses.values())]
    else:
        mult_lists = [multipliers for _ in layers]
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    markers = ['o', 'x', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
    for layer, label, mults in zip(layers, settingses, mult_lists):
        for i, behavior in enumerate(behaviors):
            print(f"{label} {behavior}")
            if rescale:
                # print('getting vec')
                vec, nvec = get_vec(behavior, layer=layer)
                lnvec, lnnvec = vec.norm(), nvec.norm()
                stdev = settings.stdev
                scale = lnvec if stdev else lnnvec
            else:
                scale = 1

            settings, color = settingses[label]
            settings.behavior = behavior

            # print('getting mprobs')
            mprobs = [get_mprobs(layer, m, settings) for m in mults]
            means = [np.mean(m) for m in mprobs]
            xs = [m * scale for m in mults]
            # print('plotting')
            ax.plot(xs, means, label=f"{label} {behavior}", color=color, marker=markers[i])
    ax.legend()
    return fig, ax

def mults_plot_grid(layers: list[int], multipliers: list[float], 
    settingses: dict[str, (SteeringSettings, str)],
    behaviors: list[str] = ALL_BEHAVIORS,
    rescale=False,
    colors=True,
    ab_sum=False,
    a_vs_b=False,
    ):
    if multipliers is None:
        mult_lists = [get_mults(layer, pair[0]) for layer, pair in zip(layers, settingses.values())]
    else:
        mult_lists = [multipliers for _ in layers]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    markers = ['o', 'x', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
    for layer, label, mults in zip(layers, settingses, mult_lists):
        for i, behavior in enumerate(behaviors):
            print(f"{label} {behavior}")
            if rescale:
                # print('getting vec')
                vec, nvec = get_vec(behavior, layer=layer)
                lnvec, lnnvec = vec.norm(), nvec.norm()
                stdev = settings.stdev
                scale = lnvec if stdev else lnnvec
            else:
                scale = 1

            settings, color = settingses[label]
            settings.behavior = behavior

            # print('getting mprobs')
            mprobs = [get_mprobs(layer, m, settings) for m in mults]
            mu_pairs = [get_probs(layer, m, settings, not a_vs_b) for m in mults]
            if ab_sum:
                mprobs = [match_probs + unmatch_probs for match_probs, unmatch_probs in mu_pairs]
            else:
                mprobs = [match_probs / (match_probs + unmatch_probs) for match_probs, unmatch_probs in mu_pairs]
            means = [np.mean(m) for m in mprobs]
            xs = [m * scale for m in mults]
            # print('plotting')
            ax = axs[i//4, i%4]
            if color is None or not colors:
                ax.plot(xs, means, label=f"{label}")
            else:
                ax.plot(xs, means, label=f"{label}", color=color)#, marker=markers[i])
            ax.set_title(behavior)
            ax.set_xlabel("CAA multiplier")
            if ab_sum:
                ax.set_ylabel("p(A) + p(B)")
            elif a_vs_b:
                ax.set_ylabel("p(A) / (p(A) + p(B))")
            else:
                ax.set_ylabel("p(matching) / (p(A) + p(B))")
            ax.legend()
            ax.set_ylim(0, 1)
    return fig, axs

def mults_sample_grid(layers: list[int], multipliers: list[float], 
    settingses: dict[str, (SteeringSettings, str)],
    behaviors: list[str] = ALL_BEHAVIORS,
    rescale=False,
    colors=True,
    ):
    if multipliers is None:
        mult_lists = [get_mults(layer, pair[0]) for layer, pair in zip(layers, settingses.values())]
    else:
        mult_lists = [multipliers for _ in layers]
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    markers = ['o', 'x', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
    for layer, label, mults in zip(layers, settingses, mult_lists):
        for i, behavior in enumerate(behaviors):
            print(f"{label} {behavior}")
            if rescale:
                # print('getting vec')
                vec, nvec = get_vec(behavior, layer=layer)
                lnvec, lnnvec = vec.norm(), nvec.norm()
                stdev = settings.stdev
                scale = lnvec if stdev else lnnvec
            else:
                scale = 1

            settings, color = settingses[label]
            settings.behavior = behavior

            # print('getting mprobs')
            mprobs = [get_mprobs(layer, m, settings) for m in mults]
            for j in range(20):
                samps = [m[j] for m in mprobs]
                xs = [m * scale for m in mults]
                # print('plotting')
                ax = axs[i//4, i%4]
                if color is None or not colors:
                    ax.plot(xs, samps, label=f"{label}", alpha=0.1)
                else:
                    ax.plot(xs, samps, label=f"{label}", color=color, alpha=0.1)#, marker=markers[i])
                ax.set_title(behavior)
                ax.set_xlabel("multiplier")
                ax.set_ylabel("matching prob")
                # ax.legend()
                ax.set_ylim(0, 1)
    return fig, axs

def sweep_plot(layers: list[int], multiplier: float, settingses: dict[str, (SteeringSettings, str)], behaviors: list[str] = ALL_BEHAVIORS):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    markers = ['o', 'x', 's', 'D', '^', 'v', '<', '>', 'p', 'h']
    for label in settingses:
        for i, behavior in enumerate(behaviors):
            settings, color = settingses[label]
            settings.behavior = behavior
            mprobs = [get_mprobs(layer, multiplier, settings) for layer in layers]
            means = [np.mean(m) for m in mprobs]
            ax.plot(layers, means, label=f"{label} {behavior}", color=color, marker=markers[i])
    ax.legend()
    plt.show()


# fig, ax = mults_plot([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False, open_response=True
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", open_response=True
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, open_response=True
#         ), 'green'),
# })

# ax.set_title("7B-chat layer 13\n(open -> MC)")
# ax.set_xlabel("multiplier")
# ax.set_ylabel("matching prob")
# ax.set_ylim(0, 1)
# # write to file
# plt.savefig("plots/open_mc.png")

# ax.set_xlim(-.25, .25)
# plt.savefig("plots/open_mc_zoom.png")



# fig, ax = mults_plot([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False, open_response=True, after_instr=False,
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", open_response=True, after_instr=False,
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, open_response=True, after_instr=False,
#         ), 'green'),
# })

# ax.set_title("7B-chat layer 13\nall toks (open -> MC)")
# ax.set_xlabel("multiplier")
# ax.set_ylabel("matching prob")
# ax.set_ylim(0, 1)
# # write to file
# plt.savefig("plots/open_mc_alltok.png")

# ax.set_xlim(-.25, .25)
# plt.savefig("plots/open_mc_alltok_zoom.png")


# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False, open_response=True
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", open_response=True
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, open_response=True
#         ), 'green'),
# },
# colors=False,
# )
# fig.suptitle("7B-chat layer 13\n(open -> MC)")
# plt.savefig("plots/open_mc_grid.png")


# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False, open_response=True
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", open_response=True
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, open_response=True
#         ), 'green'),
# },
# colors=False,
# ab_sum=True,
# )
# fig.suptitle("7B-chat layer 13: p(A) + p(B)\n(open -> MC)")
# plt.savefig("plots/open_mc_grid_sum.png")


# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False, open_response=True, after_instr=False,
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", open_response=True, after_instr=False,
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, open_response=True, after_instr=False,
#         ), 'green'),
# },
# colors=False,
# )
# fig.suptitle("7B-chat layer 13\n(open -> MC) alltok")
# plt.savefig("plots/open_mc_alltok_grid.png")


# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False,
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth",
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True,
#         ), 'green'),
# },
# colors=False,
# )
# fig.suptitle("7B-chat layer 13\n(MC -> MC)")
# plt.savefig("plots/mc_mc_grid.png")


# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False,
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth",
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True,
#         ), 'green'),
# },
# colors=False,
# ab_sum=True,
# )
# fig.suptitle("7B-chat layer 13: p(A) + p(B)\n(MC -> MC)")
# plt.savefig("plots/mc_mc_grid_sum.png")


# fig, axs = mults_sample_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False,
#         ), 'blue'), 
# },
# colors=True,
# )
# fig.suptitle("7B-chat layer 13\n(MC -> MC) CAA")
# plt.savefig("plots/mc_mc_sampgrid_caa.png")


# fig, axs = mults_sample_grid([13, 13, 13] , None, {
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth",
#         ), 'red'),
# },
# colors=True,
# )
# fig.suptitle("7B-chat layer 13\n(MC -> MC) CAA+orth")
# plt.savefig("plots/mc_mc_sampgrid_orth.png")


# fig, axs = mults_sample_grid([13, 13, 13] , None, {
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True,
#         ), 'green'),
# },
# colors=True,
# )
# fig.suptitle("7B-chat layer 13\n(MC -> MC) CAA+leace")
# plt.savefig("plots/mc_mc_sampgrid_leace.png")



# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False, open_response=True, only_instr=True,
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", open_response=True, only_instr=True,
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, open_response=True, only_instr=True,
#         ), 'green'),
# },
# colors=False,
# )
# fig.suptitle("7B-chat layer 13\n(open -> MC)")
# plt.savefig("plots/open_mc_instr_grid.png")



# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False,  only_instr=True,
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", only_instr=True,
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, only_instr=True,
#         ), 'green'),
# },
# colors=False,
# )
# fig.suptitle("7B-chat layer 13\n(open -> MC)")
# plt.savefig("plots/mc_mc_instr_grid.png")

# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "CAA": (SteeringSettings(
#         model_size='7b', normalized=False,  after_instr=False,
#         ), 'blue'), 
#     "CAA+orth": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, leace_method="orth", after_instr=False,
#         ), 'red'),
#     "CAA+leace": (SteeringSettings(
#         model_size='7b', normalized=False, leace=True, after_instr=False,
#         ), 'green'),
# },
# colors=False,
# )
# fig.suptitle("7B-chat layer 13\n(open -> MC)")
# plt.savefig("plots/mc_mc_alltok_grid.png")


# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "default": (SteeringSettings(
#         model_size='7b', normalized=False,
#         ), 'blue'), 
#     "instr": (SteeringSettings(
#         model_size='7b', normalized=False, only_instr=True,
#         ), 'red'),
#     "alltok": (SteeringSettings(
#         model_size='7b', normalized=False, after_instr=False,
#         ), 'green'),
# },
# colors=False,
# )
# fig.suptitle("7B-chat layer 13\n(MC -> MC)")
# plt.savefig("plots/mc_mc_toks_grid.png")


# fig, axs = mults_plot_grid([13, 13, 13] , None, {
#     "default": (SteeringSettings(
#         model_size='7b', normalized=False,
#         ), 'blue'), 
#     "instr": (SteeringSettings(
#         model_size='7b', normalized=False, only_instr=True,
#         ), 'red'),
#     "alltok": (SteeringSettings(
#         model_size='7b', normalized=False, after_instr=False,
#         ), 'green'),
# },
# colors=False,
# ab_sum=True,
# )
# fig.suptitle("7B-chat layer 13: p(A)+p(B)\n(MC -> MC)")
# plt.savefig("plots/mc_mc_toks_sum_grid.png")


fig, axs = mults_plot_grid([13]*6 , None, {
    "default": (SteeringSettings(
        model_size='7b', normalized=False,
        ), 'blue'), 
    "instr": (SteeringSettings(
        model_size='7b', normalized=False, only_instr=True,
        ), 'red'),
    "alltok": (SteeringSettings(
        model_size='7b', normalized=False, after_instr=False,
        ), 'green'),
    "logit": (SteeringSettings(
        model_size='7b', normalized=False, logit=True,
        ), 'green'),
    "leace": (SteeringSettings(
        model_size='7b', normalized=False, leace=True,
        ), 'green'),
    "open": (SteeringSettings(
        model_size='7b', normalized=False, leace=True, open_response=True,
        ), 'green'),
},
colors=False,
a_vs_b=True,
)
fig.suptitle("7B-chat layer 13: A vs B\n(mostly-MC -> MC)")
plt.savefig("plots/mc_mc_toks_ab.png")



fig, axs = mults_plot_grid([13]*5 , None, {
    "default": (SteeringSettings(
        model_size='7b', normalized=False, balanced=True,
        ), 'blue'), 
    "instr": (SteeringSettings(
        model_size='7b', normalized=False, only_instr=True, balanced=True,
        ), 'red'),
    "alltok": (SteeringSettings(
        model_size='7b', normalized=False, after_instr=False, balanced=True,
        ), 'green'),
    "logit": (SteeringSettings(
        model_size='7b', normalized=False, logit=True, balanced=True,
        ), 'green'),
    "leace": (SteeringSettings(
        model_size='7b', normalized=False, leace=True, balanced=True,
        ), 'green'),
},
colors=False,
a_vs_b=True,
)
fig.suptitle("7B-chat layer 13: A vs B\n(BalancedMC -> MC)")
plt.savefig("plots/bal_mc_toks_ab.png")

fig, axs = mults_plot_grid([13]*5 , None, {
    "default": (SteeringSettings(
        model_size='7b', normalized=False, balanced=True,
        ), 'blue'), 
    "instr": (SteeringSettings(
        model_size='7b', normalized=False, only_instr=True, balanced=True,
        ), 'red'),
    "alltok": (SteeringSettings(
        model_size='7b', normalized=False, after_instr=False, balanced=True,
        ), 'green'),
    "logit": (SteeringSettings(
        model_size='7b', normalized=False, logit=True, balanced=True,
        ), 'green'),
    "leace": (SteeringSettings(
        model_size='7b', normalized=False, leace=True, balanced=True,
        ), 'green'),
},
colors=False,
)
fig.suptitle("7B-chat layer 13: p(Matching)\n(BalancedMC -> MC)")
plt.savefig("plots/bal_mc_toks_grid.png")