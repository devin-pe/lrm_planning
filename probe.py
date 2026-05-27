#!/usr/bin/env python3
"""Probe Qwen hidden states for ToH Sierpinski structure using prefill-only activations.

Pipeline:
1) Enumerate all 81 states for 4-disk ToH and build prompts from shared prompt logic.
2) Run prompt-only forward passes and extract last-prompt-token hidden states.
3) Build ToH graph distances (all-pairs shortest path).
4) Train per-layer 2D linear probes with pairwise distance-matching loss.
5) Evaluate layers (Spearman, nearest-adjacent accuracy, stress) and visualize.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from prompts import create_nonstandard_prompt

State = Tuple[int, int, int, int]
Edge = Tuple[int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe Qwen hidden states for ToH geometry")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-27B")
    parser.add_argument("--n_disks", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="outputs/qwen_probe")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trained_transformer_best_layer", type=int, default=5)
    parser.add_argument("--trained_transformer_spearman", type=float, default=0.93)
    parser.add_argument("--trained_transformer_top1_adj_acc", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_arg(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def enumerate_states(n_disks: int) -> List[State]:
    return [tuple(int(v) for v in s) for s in product(range(3), repeat=n_disks)]


def state_tuple_to_pegs(state: State, n_disks: int) -> List[List[int]]:
    pegs = [[], [], []]
    for disk in range(n_disks, 0, -1):
        peg = int(state[disk - 1])
        pegs[peg].append(disk)
    return pegs


def state_to_label(state: State) -> str:
    return "".join(str(x) for x in state)


def build_prompts(tokenizer: AutoTokenizer, states: Sequence[State], n_disks: int) -> Tuple[List[str], List[List[int]]]:
    prompts: List[str] = []
    goal_pegs = [[], [], list(range(n_disks, 0, -1))]

    for i, st in enumerate(states):
        pegs = state_tuple_to_pegs(st, n_disks=n_disks)
        system_prompt, user_prompt, _ = create_nonstandard_prompt(
            num_disks=n_disks,
            problem_id=i,
            seed=0,
            initial_state_override=pegs,
            goal_state_override=goal_pegs,
        )
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt)

    return prompts, goal_pegs


def top_disk_per_peg(state: State, n_disks: int) -> List[int | None]:
    tops: List[int | None] = [None, None, None]
    for disk in range(1, n_disks + 1):
        peg = state[disk - 1]
        if tops[peg] is None:
            tops[peg] = disk
    return tops


def legal_neighbors(state: State, n_disks: int) -> List[State]:
    tops = top_disk_per_peg(state, n_disks)
    out: List[State] = []
    for src in range(3):
        src_top = tops[src]
        if src_top is None:
            continue
        for dst in range(3):
            if src == dst:
                continue
            dst_top = tops[dst]
            if dst_top is None or src_top < dst_top:
                nxt = list(state)
                nxt[src_top - 1] = dst
                out.append(tuple(int(x) for x in nxt))
    return out


def build_graph_and_distances(states: Sequence[State], n_disks: int) -> Tuple[np.ndarray, List[Edge], Dict[int, Set[int]]]:
    n = len(states)
    state_to_idx = {s: i for i, s in enumerate(states)}

    adjacency: List[List[int]] = [[] for _ in range(n)]
    edge_set: Set[Tuple[int, int]] = set()
    neighbor_sets: Dict[int, Set[int]] = {i: set() for i in range(n)}

    for i, s in enumerate(states):
        for nbr in legal_neighbors(s, n_disks=n_disks):
            j = state_to_idx[nbr]
            adjacency[i].append(j)
            neighbor_sets[i].add(j)
            a, b = sorted((i, j))
            edge_set.add((a, b))

    dist = np.full((n, n), np.inf, dtype=np.float32)
    for src in range(n):
        dist[src, src] = 0.0
        q = deque([src])
        while q:
            u = q.popleft()
            for v in adjacency[u]:
                if np.isinf(dist[src, v]):
                    dist[src, v] = dist[src, u] + 1.0
                    q.append(v)

    if np.isinf(dist).any():
        raise RuntimeError("Found disconnected states in ToH graph")

    return dist, sorted(edge_set), neighbor_sets


def select_layers(n_layers_total: int) -> List[int]:
    # 1-indexed layer ids for reporting/saving consistency.
    layers = {1, n_layers_total}
    layers.update(range(4, n_layers_total + 1, 4))
    return sorted(layers)


def _find_layers_list(model: nn.Module):
    """Return the transformer decoder layers list regardless of model wrapper."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise RuntimeError(
        "Cannot locate transformer layers. Expected model.model.layers or model.layers."
    )


def extract_hidden_states_last_prompt_token(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    prompts: Sequence[str],
    layer_ids_1based: Sequence[int],
    device: torch.device,
    batch_size: int,
    hidden_size: int,
) -> Dict[int, torch.Tensor]:
    n_states = len(prompts)
    per_layer: Dict[int, torch.Tensor] = {
        l: torch.empty((n_states, hidden_size), dtype=torch.float32) for l in layer_ids_1based
    }

    layers_list = _find_layers_list(model)
    captured: Dict[int, torch.Tensor] = {}

    def make_hook(lid: int):
        def hook_fn(module, input, output):
            h = output[0] if isinstance(output, tuple) else output
            # Move to CPU immediately to free GPU memory; keep full batch for now.
            captured[lid] = h.detach().float().cpu()
        return hook_fn

    hooks = []
    for l in layer_ids_1based:
        h = layers_list[l - 1].register_forward_hook(make_hook(l))
        hooks.append(h)

    i = 0
    cur_bs = max(1, int(batch_size))
    try:
        while i < n_states:
            batch_prompts = list(prompts[i : i + cur_bs])
            try:
                enc = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=False)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                captured.clear()
                with torch.no_grad():
                    model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )

                seq_len = attention_mask.shape[1]
                # Index of last real (non-padding) token per sequence.
                lengths = (attention_mask.cpu() * torch.arange(seq_len)).argmax(dim=1)
                for b in range(input_ids.shape[0]):
                    last_idx = int(lengths[b].item())
                    global_i = i + b
                    for l in layer_ids_1based:
                        per_layer[l][global_i] = captured[l][b, last_idx, :]

                i += input_ids.shape[0]
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and cur_bs > 1:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    cur_bs = max(1, cur_bs // 2)
                    print(f"[WARN] OOM during extraction; reducing batch_size to {cur_bs} and retrying.")
                    continue
                raise
    finally:
        for h in hooks:
            h.remove()

    return per_layer


def normalize_distances(dist: np.ndarray) -> Tuple[np.ndarray, float, float]:
    mu = float(dist.mean())
    sigma = float(dist.std())
    if sigma < 1e-8:
        sigma = 1.0
    return ((dist - mu) / sigma).astype(np.float32), mu, sigma


def train_distance_probe(
    x: torch.Tensor,
    true_dist_norm: torch.Tensor,
    epochs: int,
    lr: float,
    device: torch.device,
) -> Tuple[nn.Linear, np.ndarray, float]:
    x = x.to(device)
    y_dist = true_dist_norm.to(device)

    probe = nn.Linear(x.shape[1], 2, bias=True).to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)

    final_loss = float("nan")
    for ep in range(1, epochs + 1):
        opt.zero_grad(set_to_none=True)
        z = probe(x)
        pred_dist = torch.cdist(z, z, p=2)
        loss = torch.mean((pred_dist - y_dist) ** 2)
        loss.backward()
        opt.step()
        final_loss = float(loss.item())

        if ep == 1 or ep % 200 == 0 or ep == epochs:
            print(f"[probe] epoch={ep:4d} loss={final_loss:.8f}")

    with torch.no_grad():
        coords = probe(x).detach().cpu().numpy().astype(np.float32)

    return probe, coords, final_loss


def evaluate_layer(
    coords_2d: np.ndarray,
    true_dist: np.ndarray,
    neighbor_sets: Dict[int, Set[int]],
) -> Tuple[float, float, float, float, float]:
    pred_dist = np.linalg.norm(coords_2d[:, None, :] - coords_2d[None, :, :], axis=2)

    triu_i, triu_j = np.triu_indices(pred_dist.shape[0], k=1)
    pred_pairs = pred_dist[triu_i, triu_j]
    true_pairs = true_dist[triu_i, triu_j]
    rho, _ = spearmanr(pred_pairs, true_pairs)
    pc = pred_pairs.astype(np.float64) - pred_pairs.mean()
    tc = true_pairs.astype(np.float64) - true_pairs.mean()
    denom_p = float(np.sqrt(np.sum(pc * pc) * np.sum(tc * tc)))
    pearson = float(np.sum(pc * tc) / denom_p) if denom_p > 0 else 0.0

    n = pred_dist.shape[0]
    correct = 0
    per_state_mean_ranks: List[float] = []
    for i in range(n):
        row = pred_dist[i].copy()
        row[i] = np.inf
        nn_idx = int(np.argmin(row))
        if nn_idx in neighbor_sets[i]:
            correct += 1
        ranks = np.argsort(np.argsort(row))
        nbrs = neighbor_sets[i]
        if nbrs:
            per_state_mean_ranks.append(float(np.mean([ranks[j] + 1 for j in nbrs])))

    nearest_state_acc = correct / n
    mean_neighbor_rank = float(np.mean(per_state_mean_ranks)) if per_state_mean_ranks else float("nan")

    numer = float(np.sum((pred_dist - true_dist) ** 2))
    denom = float(np.sum(true_dist**2))
    stress = math.sqrt(numer / denom) if denom > 1e-12 else 0.0

    return float(rho), float(nearest_state_acc), float(stress), mean_neighbor_rank, pearson


def plot_layer_geometry(
    coords: np.ndarray,
    states: Sequence[State],
    edges: Sequence[Edge],
    title: str,
    out_path: Path,
    color_by_disk_idx: int,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)

    for i, j in edges:
        ax.plot(
            [coords[i, 0], coords[j, 0]],
            [coords[i, 1], coords[j, 1]],
            color="gray",
            linewidth=0.8,
            alpha=0.5,
            zorder=1,
        )

    palette = {0: "red", 1: "blue", 2: "green"}
    color_vals = np.array([s[color_by_disk_idx] for s in states], dtype=np.int64)

    for peg in [0, 1, 2]:
        mask = color_vals == peg
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=80,
            color=palette[peg],
            alpha=0.95,
            zorder=2,
            label=f"disk[{color_by_disk_idx}] on peg {peg}",
        )

    for i, st in enumerate(states):
        ax.text(
            coords[i, 0],
            coords[i, 1],
            state_to_label(st),
            fontsize=6,
            ha="center",
            va="center",
            bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            zorder=3,
        )

    ax.set_title(title)
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_spearman_bar(layer_ids: Sequence[int], spearmans: Sequence[float], best_layer: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    colors = ["tab:orange" if l == best_layer else "tab:blue" for l in layer_ids]
    ax.bar([str(l) for l in layer_ids], spearmans, color=colors)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Spearman rho")
    ax.set_title("Qwen3-27B: Which Layer Encodes the State Space?")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def print_layer_table(rows: Sequence[Dict[str, float]]) -> None:
    print("\nLayer | Spearman rho | Nearest-State Acc | Mean Nbr Rank | Stress | Probe Loss")
    print("------+--------------+-------------------+---------------+--------+-----------")
    for r in rows:
        print(
            f"{int(r['layer']):5d} | {r['spearman_rho']:12.4f} | {r['nearest_state_acc']:17.4f} | "
            f"{r['mean_neighbor_rank']:13.2f} | {r['stress']:6.4f} | {r['probe_loss']:.6f}"
        )


def print_comparison(
    best_layer: int,
    best_spearman: float,
    best_nearest_state_acc: float,
    tr_layer: int,
    tr_spearman: float,
    tr_top1_adj_acc: float,
    model_name: str = "LLM (no fine-tune)",
) -> None:
    label = model_name.split("/")[-1]  # strip HF org or path prefix
    col_w = max(len(label), len("Trained ToH Transformer"), 24)
    sep = "-" * col_w
    print(f"\n{'Model':<{col_w}} | Best Layer | Spearman rho | Nearest-State Acc")
    print(f"{sep}-+------------+--------------+------------------")
    print(f"{'Trained ToH Transformer':<{col_w}} | {tr_layer:10d} | {tr_spearman:12.4f} | {tr_top1_adj_acc:16.4f}")
    print(f"{label:<{col_w}} | {best_layer:10d} | {best_spearman:12.4f} | {best_nearest_state_acc:16.4f}")


def main() -> None:
    args = parse_args()
    if args.n_disks != 4:
        raise ValueError("This script is configured for 4 disks")

    set_seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    states = enumerate_states(args.n_disks)
    if len(states) != 81:
        raise RuntimeError("Expected 81 states for 4 disks")

    print("[INFO] Loading tokenizer/model")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _load_kw = dict(
        torch_dtype=dtype_from_arg(args.dtype),
        trust_remote_code=True,
        device_map="auto" if args.device == "cuda" else None,
    )
    try:
        _raw = AutoModelForCausalLM.from_pretrained(args.model_name, **_load_kw)
    except (ValueError, KeyError, AttributeError):
        print("[INFO] AutoModelForCausalLM failed; loading as AutoModel (VLM path)")
        _raw = AutoModel.from_pretrained(args.model_name, **_load_kw)

    # For VLMs (e.g. Qwen3.5-VL) the text transformer lives at model.model.language_model.
    # Text-only probing needs just that submodule; vision encoder is irrelevant.
    if hasattr(_raw, "model") and hasattr(_raw.model, "language_model"):
        print("[INFO] VLM detected — probing model.model.language_model")
        model = _raw.model.language_model
        lm_cfg = getattr(_raw.config, "text_config", _raw.config)
    else:
        model = _raw
        lm_cfg = _raw.config

    device = torch.device(args.device)
    if args.device != "cuda":
        model.to(device)
    model.eval()

    print("[INFO] Building prompts")
    prompts, goal_pegs = build_prompts(tokenizer, states, args.n_disks)

    n_layers_total = int(lm_cfg.num_hidden_layers)
    hidden_size = int(lm_cfg.hidden_size)
    layer_ids = select_layers(n_layers_total)
    print(f"[INFO] num_hidden_layers={n_layers_total}")
    print(f"[INFO] extracting layers={layer_ids}")

    print("[INFO] Extracting prefill hidden states at last prompt token")
    hidden_by_layer = extract_hidden_states_last_prompt_token(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        layer_ids_1based=layer_ids,
        device=device,
        batch_size=args.batch_size,
        hidden_size=hidden_size,
    )

    hidden_path = out_dir / "hidden_states.pt"
    torch.save(
        {
            "model_name": args.model_name,
            "layer_ids": layer_ids,
            "states": [list(s) for s in states],
            "goal_state": goal_pegs,
            "hidden_states": hidden_by_layer,
        },
        hidden_path,
    )

    print("[INFO] Building graph distances")
    true_dist, edges, neighbor_sets = build_graph_and_distances(states, args.n_disks)
    true_dist_norm, dist_mu, dist_std = normalize_distances(true_dist)

    y_norm_t = torch.tensor(true_dist_norm, dtype=torch.float32)

    print("[INFO] Training distance-matching probes")
    results: List[Dict[str, float]] = []
    per_layer_coords: Dict[int, np.ndarray] = {}

    for layer in layer_ids:
        print(f"\n[INFO] Layer {layer}")
        x_t = hidden_by_layer[layer]
        probe, coords, probe_loss = train_distance_probe(
            x=x_t,
            true_dist_norm=y_norm_t,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
        )

        rho, nearest_state_acc, stress, mean_neighbor_rank, pearson = evaluate_layer(coords, true_dist, neighbor_sets)
        per_layer_coords[layer] = coords

        torch.save(
            {
                "layer": layer,
                "state_dict": probe.state_dict(),
                "probe_loss": float(probe_loss),
                "spearman_rho": float(rho),
                "pearson_r": float(pearson),
                "nearest_state_acc": float(nearest_state_acc),
                "mean_neighbor_rank": float(mean_neighbor_rank),
                "stress": float(stress),
            },
            out_dir / f"probe_layer_{layer}.pt",
        )

        title = f"Qwen3-27B - Layer {layer} - Spearman={rho:.3f}"
        plot_layer_geometry(
            coords=coords,
            states=states,
            edges=edges,
            title=title,
            out_path=out_dir / f"layer_{layer:02d}_sierpinski_largest_disk.png",
            color_by_disk_idx=3,
        )

        results.append(
            {
                "layer": float(layer),
                "spearman_rho": float(rho),
                "pearson_r": float(pearson),
                "nearest_state_acc": float(nearest_state_acc),
                "mean_neighbor_rank": float(mean_neighbor_rank),
                "stress": float(stress),
                "probe_loss": float(probe_loss),
            }
        )

    results_sorted = sorted(results, key=lambda r: r["spearman_rho"], reverse=True)
    print_layer_table(results_sorted)

    best_layer = int(results_sorted[0]["layer"])
    best_spearman = float(results_sorted[0]["spearman_rho"])
    best_top1 = float(results_sorted[0]["nearest_state_acc"])

    plot_layer_geometry(
        coords=per_layer_coords[best_layer],
        states=states,
        edges=edges,
        title=f"Qwen3-27B - Layer {best_layer} - second-largest disk coloring",
        out_path=out_dir / f"layer_{best_layer:02d}_sierpinski_second_largest_disk.png",
        color_by_disk_idx=2,
    )

    bar_layers = [int(r["layer"]) for r in results]
    bar_spearman = [float(r["spearman_rho"]) for r in results]
    plot_spearman_bar(bar_layers, bar_spearman, best_layer, out_dir / "spearman_by_layer.png")

    print_comparison(
        best_layer=best_layer,
        best_spearman=best_spearman,
        best_nearest_state_acc=best_top1,
        tr_layer=args.trained_transformer_best_layer,
        tr_spearman=args.trained_transformer_spearman,
        tr_top1_adj_acc=args.trained_transformer_top1_adj_acc,
        model_name=args.model_name,
    )

    summary = {
        "model_name": args.model_name,
        "n_disks": args.n_disks,
        "num_states": len(states),
        "goal_state": goal_pegs,
        "layers": layer_ids,
        "distance_normalization": {"mean": dist_mu, "std": dist_std},
        "metrics_sorted_by_spearman": results_sorted,
        "best_layer": best_layer,
        "comparison": {
            "trained_toh_transformer": {
                "best_layer": args.trained_transformer_best_layer,
                "spearman_rho": args.trained_transformer_spearman,
                "nearest_state_acc": args.trained_transformer_top1_adj_acc,
            },
            "qwen3_27b": {
                "best_layer": best_layer,
                "spearman_rho": best_spearman,
                "nearest_state_acc": best_top1,
            },
        },
        "artifacts": {
            "hidden_states": str(hidden_path),
            "probe_layer_files": [str(out_dir / f"probe_layer_{l}.pt") for l in layer_ids],
            "spearman_bar": str(out_dir / "spearman_by_layer.png"),
            "runtime_seconds": time.time() - t0,
        },
    }
    (out_dir / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n[INFO] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
