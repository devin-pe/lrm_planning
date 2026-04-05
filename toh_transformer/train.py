"""Training script for decoder-only ToH transformer."""

from __future__ import annotations

import math
import os
import random
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parent.parent))

from toh_transformer.config import TrainConfig, parse_train_args
from toh_transformer.data import (
    DatasetBundle,
    ToHFlatDataset,
    build_datasets,
    build_multi_n_datasets,
    build_tower_to_tower_multi_n_datasets,
)
from toh_transformer.model import ToHTransformer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masked_ce_and_accuracy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    loss_mask: torch.Tensor,
) -> Tuple[torch.Tensor, float, int]:
    vocab_size = logits.size(-1)
    losses = F.cross_entropy(
        logits.reshape(-1, vocab_size),
        target_ids.reshape(-1),
        reduction="none",
    ).view_as(target_ids)

    mask_f = loss_mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    loss = (losses * mask_f).sum() / denom

    preds = logits.argmax(dim=-1)
    correct = ((preds == target_ids) & loss_mask).sum().item()
    total = int(loss_mask.sum().item())
    token_acc = float(correct / max(total, 1))
    return loss, token_acc, total


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_ratio: float,
    base_lr: float,
    min_lr: float,
) -> LambdaLR:
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            current_lr = base_lr * float(step + 1) / float(warmup_steps)
        else:
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            current_lr = min_lr + (base_lr - min_lr) * cosine
        return current_lr / base_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


@torch.no_grad()
def greedy_decode_moves(
    model: ToHTransformer,
    prefix_ids: list[int],
    max_seq_len: int,
    eos_id: int,
    device: torch.device,
) -> list[int]:
    model.eval()
    seq = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
    generated: list[int] = []

    for _ in range(max_seq_len - len(prefix_ids)):
        logits = model(seq)
        next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
        generated.append(next_id)

        next_token = torch.tensor([[next_id]], dtype=torch.long, device=device)
        seq = torch.cat([seq, next_token], dim=1)

        if next_id == eos_id:
            break

    return generated


@torch.no_grad()
def sequence_accuracy(
    model: ToHTransformer,
    dataset: ToHFlatDataset,
    eos_id: int,
    max_seq_len: int,
    device: torch.device,
    max_examples: int = -1,
) -> float:
    if len(dataset) == 0:
        return 0.0

    model.eval()
    correct = 0
    encoded = dataset.encoded
    if max_examples > 0:
        encoded = encoded[:max_examples]

    for ex in encoded:
        pred = greedy_decode_moves(
            model=model,
            prefix_ids=ex.prefix_ids,
            max_seq_len=max_seq_len,
            eos_id=eos_id,
            device=device,
        )
        if pred == ex.move_target_ids:
            correct += 1

    return correct / len(encoded)


def format_metric(value: float) -> str:
    if not math.isfinite(value):
        return "n/a"
    return f"{value:.4f}"


def run_epoch(
    model: ToHTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[LambdaLR] = None,
    grad_clip: float = 1.0,
) -> Tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0

    for batch_idx, (input_ids, target_ids, loss_mask) in enumerate(loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        loss_mask = loss_mask.to(device)

        logits = model(input_ids)
        loss, batch_token_acc, batch_tokens = masked_ce_and_accuracy(logits, target_ids, loss_mask)

        if not torch.isfinite(loss):
            phase = "train" if is_train else "eval"
            raise RuntimeError(f"Non-finite loss detected during {phase} at batch {batch_idx}")

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        total_loss += float(loss.item()) * batch_tokens
        total_correct += batch_token_acc * batch_tokens
        total_tokens += batch_tokens

    mean_loss = total_loss / max(total_tokens, 1)
    token_acc = total_correct / max(total_tokens, 1)
    return mean_loss, token_acc


def save_checkpoint(
    path: Path,
    model: ToHTransformer,
    optimizer: AdamW,
    scheduler: LambdaLR,
    cfg: TrainConfig,
    epoch: int,
    metrics: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "config": cfg.__dict__,
            "metrics": metrics,
        },
        path,
    )


def main() -> None:
    cfg = parse_train_args()
    set_seed(cfg.seed)

    use_cuda = cfg.device.startswith("cuda") and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if (
        cfg.data_mode == "flat-to-flat"
        and len(cfg.train_n_disks) == 1
        and len(cfg.test_n_disks) == 0
        and cfg.train_n_disks[0] == cfg.n_disks
    ):
        print(f"[INFO] Building dataset for n_disks={cfg.n_disks}...")
        data: DatasetBundle = build_datasets(n_disks=cfg.n_disks, seed=cfg.seed)
    else:
        if cfg.data_mode == "flat-to-flat":
            print(
                "[INFO] Building mixed-n flat-to-flat dataset "
                f"train_n_disks={cfg.train_n_disks}, test_n_disks={cfg.test_n_disks}, val_ratio={cfg.val_ratio:.3f}"
            )
            data = build_multi_n_datasets(
                train_n_disks=cfg.train_n_disks,
                test_n_disks=cfg.test_n_disks,
                max_seq_len=cfg.max_seq_len,
                seed=cfg.seed,
                val_ratio=cfg.val_ratio,
            )
        else:
            print(
                "[INFO] Building mixed-n tower-to-tower dataset "
                f"train_n_disks={cfg.train_n_disks}, test_n_disks={cfg.test_n_disks}, "
                f"val_ratio={cfg.val_ratio:.3f}, strategy={cfg.tower_data_strategy}, "
                f"train_repeats={cfg.tower_train_repeats}, test_repeats={cfg.tower_test_repeats}"
            )
            data = build_tower_to_tower_multi_n_datasets(
                train_n_disks=cfg.train_n_disks,
                test_n_disks=cfg.test_n_disks,
                max_seq_len=cfg.max_seq_len,
                seed=cfg.seed,
                val_ratio=cfg.val_ratio,
                train_repeats=cfg.tower_train_repeats,
                test_repeats=cfg.tower_test_repeats,
                data_strategy=cfg.tower_data_strategy,
            )
    train_ds = data.train_dataset
    val_ds = data.val_dataset
    test_ds = data.test_dataset

    print(f"[INFO] Train examples: {len(train_ds)}")
    if val_ds is not None:
        print(f"[INFO] Val examples: {len(val_ds)}")
    if test_ds is not None:
        print(f"[INFO] Test examples: {len(test_ds)}")
    seq_eval_scope = "full split" if cfg.seq_eval_max_examples <= 0 else str(cfg.seq_eval_max_examples)
    train_seq_eval_scope = (
        seq_eval_scope
        if cfg.seq_eval_max_train_examples <= 0
        else str(cfg.seq_eval_max_train_examples)
    )
    print(
        "[INFO] Sequence eval: "
        f"max_examples_per_split={seq_eval_scope}, "
        f"max_train_examples={train_seq_eval_scope}, "
        f"include_train={cfg.seq_eval_include_train}"
    )
    print("[INFO] Test split evaluation policy: final epoch only")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=use_cuda,
    )

    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=use_cuda,
        )

    test_loader = None
    if test_ds is not None:
        test_loader = DataLoader(
            test_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=use_cuda,
        )

    model = ToHTransformer(
        vocab_size=len(data.vocab),
        max_seq_len=data.max_seq_len,
        n_layers=cfg.n_layers,
        n_heads=cfg.n_heads,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.num_epochs * max(1, len(train_loader))
    scheduler = create_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_ratio=cfg.warmup_ratio,
        base_lr=cfg.lr,
        min_lr=cfg.min_lr,
    )

    if len(cfg.train_n_disks) == 1 and len(cfg.test_n_disks) == 0 and cfg.data_mode == "flat-to-flat":
        ckpt_subdir = f"n{cfg.n_disks}"
    else:
        train_tag = "-".join(str(n) for n in cfg.train_n_disks)
        test_tag = "-".join(str(n) for n in cfg.test_n_disks) if cfg.test_n_disks else "none"
        mode_tag = "tower" if cfg.data_mode == "tower-to-tower" else "flat"
        ckpt_subdir = f"{mode_tag}_train_{train_tag}__test_{test_tag}"
    ckpt_dir = Path(cfg.checkpoint_dir) / ckpt_subdir
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_metric = -1.0
    start_time = time.time()

    for epoch in range(1, cfg.num_epochs + 1):
        do_val_eval = (epoch % cfg.eval_every) == 0 or epoch == cfg.num_epochs
        do_test_eval = epoch == cfg.num_epochs

        train_loss, train_tok_acc = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            grad_clip=cfg.grad_clip,
        )

        if do_val_eval and val_loader is not None:
            val_loss, val_tok_acc = run_epoch(model=model, loader=val_loader, device=device)
        else:
            val_loss, val_tok_acc = float("nan"), float("nan")

        if do_test_eval and test_loader is not None:
            test_loss, test_tok_acc = run_epoch(model=model, loader=test_loader, device=device)
        else:
            test_loss, test_tok_acc = float("nan"), float("nan")

        if do_val_eval:
            if cfg.seq_eval_include_train:
                train_seq_max_examples = (
                    cfg.seq_eval_max_examples
                    if cfg.seq_eval_max_train_examples <= 0
                    else cfg.seq_eval_max_train_examples
                )
                train_seq_acc = sequence_accuracy(
                    model=model,
                    dataset=train_ds,
                    eos_id=data.vocab.eos_id,
                    max_seq_len=data.max_seq_len,
                    device=device,
                    max_examples=train_seq_max_examples,
                )
            else:
                train_seq_acc = float("nan")

            if val_ds is not None:
                val_seq_acc = sequence_accuracy(
                    model=model,
                    dataset=val_ds,
                    eos_id=data.vocab.eos_id,
                    max_seq_len=data.max_seq_len,
                    device=device,
                    max_examples=cfg.seq_eval_max_examples,
                )
                score = val_seq_acc
            else:
                val_seq_acc = float("nan")
                score = train_seq_acc if cfg.seq_eval_include_train else train_tok_acc

            if do_test_eval and test_ds is not None:
                test_seq_acc = sequence_accuracy(
                    model=model,
                    dataset=test_ds,
                    eos_id=data.vocab.eos_id,
                    max_seq_len=data.max_seq_len,
                    device=device,
                    max_examples=cfg.seq_eval_max_examples,
                )
            else:
                test_seq_acc = float("nan")
        else:
            train_seq_acc = float("nan")
            val_seq_acc = float("nan")
            test_seq_acc = float("nan")
            score = None

        elapsed = time.time() - start_time
        lr_now = scheduler.get_last_lr()[0]
        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={format_metric(train_loss)} train_tok_acc={format_metric(train_tok_acc)} "
            f"train_seq_acc={format_metric(train_seq_acc)} "
            f"val_loss={format_metric(val_loss)} val_tok_acc={format_metric(val_tok_acc)} "
            f"val_seq_acc={format_metric(val_seq_acc)} "
            f"test_loss={format_metric(test_loss)} test_tok_acc={format_metric(test_tok_acc)} "
            f"test_seq_acc={format_metric(test_seq_acc)} "
            f"lr={lr_now:.6e} elapsed={elapsed:.1f}s"
        )

        metrics = {
            "train_loss": train_loss,
            "train_tok_acc": train_tok_acc,
            "train_seq_acc": train_seq_acc,
            "val_loss": val_loss,
            "val_tok_acc": val_tok_acc,
            "val_seq_acc": val_seq_acc,
            "test_loss": test_loss,
            "test_tok_acc": test_tok_acc,
            "test_seq_acc": test_seq_acc,
            "lr": lr_now,
        }

        if score is not None and score > best_metric:
            best_metric = score
            save_checkpoint(
                path=ckpt_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                metrics=metrics,
            )

        if epoch % cfg.save_every == 0:
            save_checkpoint(
                path=ckpt_dir / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                cfg=cfg,
                epoch=epoch,
                metrics=metrics,
            )

    print("[INFO] Training complete.")
    if len(cfg.train_n_disks) == 1 and cfg.train_n_disks[0] == 3:
        print(f"[INFO] Success target: sequence accuracy >= 0.99 (best={best_metric:.4f})")
    else:
        print(f"[INFO] Success target: sequence accuracy >= 0.95 (best={best_metric:.4f})")


if __name__ == "__main__":
    main()
