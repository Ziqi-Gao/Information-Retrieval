import argparse
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup

from .data import RLHNRetrievalDataset, collate_fn
from .experiments import get_version_spec, version_names
from .losses import final_loop_loss, loopwise_loss, standard_loss
from .model import LoopMatryoshkaRetriever
from .utils import (
    add_bool_arg,
    append_jsonl,
    current_lr,
    ensure_dir,
    load_yaml,
    merge_cli_with_config,
    set_seed,
    tensor_to_float,
)


DEFAULTS: Dict[str, Any] = {
    "model_name_or_path": "answerdotai/ModernBERT-base",
    "dataset_name": "rlhn/rlhn-680K",
    "version": "standard",
    "output_dir": "outputs/standard",
    "tmax": 10,
    "num_negatives": 7,
    "max_query_length": 128,
    "max_doc_length": 512,
    "loop_impl": "memory_token",
    "detach_memory": False,
    "train_sample_size": None,
    "epochs": 1,
    "batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate_encoder": 2e-5,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "tau": 0.05,
    "bf16": False,
    "fp16": False,
    "seed": 42,
    "log_steps": 10,
    "save_steps": 1000,
    "max_steps": None,
    "dataloader_num_workers": 4,
    "use_inbatch": False,
    "loop_memory_mode": "mean_pool",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train loop-wise Matryoshka retrieval models.")
    parser.add_argument("--config", default=None, help="YAML config path. CLI overrides take precedence.")
    parser.add_argument("--model_name_or_path", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--version", choices=version_names(), default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--tmax", type=int, default=None)
    parser.add_argument("--num_negatives", type=int, default=None)
    parser.add_argument("--max_query_length", type=int, default=None)
    parser.add_argument("--max_doc_length", type=int, default=None)
    parser.add_argument("--loop_impl", default=None)
    add_bool_arg(parser, "detach_memory", "Detach previous query hidden states before passing them to the next loop.")
    parser.add_argument("--train_sample_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate_encoder", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=None)
    parser.add_argument("--tau", type=float, default=None)
    add_bool_arg(parser, "bf16", "Use bfloat16 autocast on CUDA.")
    add_bool_arg(parser, "fp16", "Use float16 autocast on CUDA.")
    add_bool_arg(parser, "use_inbatch", "Compatibility flag; training objective is hard-negative only.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--dataloader_num_workers", type=int, default=None)
    parser.add_argument("--loop_memory_mode", choices=["first_token", "mean_pool", "token_concat"], default=None)
    return parser


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser()
    raw_args = parser.parse_args()
    config = load_yaml(raw_args.config)
    args = merge_cli_with_config(raw_args, DEFAULTS, config)
    if args.bf16 and args.fp16:
        raise ValueError("Choose only one mixed precision mode: bf16 or fp16.")
    if args.use_inbatch:
        print("use_inbatch=True was requested, but this experiment uses hard-negative loss only. Forcing use_inbatch=False.")
        args.use_inbatch = False
    return args


def build_optimizer(model: LoopMatryoshkaRetriever, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.learning_rate_encoder, "name": "encoder"},
        ],
        weight_decay=args.weight_decay,
    )


def assert_encoder_only_trainable(model: LoopMatryoshkaRetriever) -> None:
    unexpected = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("encoder.")
    ]
    if unexpected:
        preview = ", ".join(unexpected[:8])
        raise AssertionError(f"Only encoder parameters may be trainable; unexpected trainable params: {preview}")


def select_loss(
    version: str,
    q_loops: list[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    loss_type = get_version_spec(version).loss_type
    if loss_type == "standard":
        return standard_loss(q_loops[0], pos_emb, neg_emb, tau=args.tau, use_inbatch=args.use_inbatch)
    if loss_type == "final_loop":
        return final_loop_loss(q_loops, pos_emb, neg_emb, tau=args.tau, use_inbatch=args.use_inbatch)
    if loss_type == "loopwise":
        return loopwise_loss(q_loops, pos_emb, neg_emb, tau=args.tau, use_inbatch=args.use_inbatch)
    raise ValueError(f"Unknown loss_type {loss_type!r} for version: {version}")


def encode_training_batch(
    model: LoopMatryoshkaRetriever,
    batch: Dict[str, Any],
    version: str,
) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
    if not get_version_spec(version).is_standard_family:
        return model.forward_batch(batch["queries"], batch["positives"], batch["negatives"])

    queries = batch["queries"]
    positives = batch["positives"]
    negatives = batch["negatives"]
    if len(queries) != len(positives) or len(queries) != len(negatives):
        raise ValueError("queries, positives, and negatives must have the same batch size.")
    batch_size = len(queries)
    num_negatives = len(negatives[0])
    if any(len(row) != num_negatives for row in negatives):
        raise ValueError("Every sample must have the same number of negatives.")

    device = model._device()
    q_emb = model.encode_queries(queries, batch_size=batch_size, loop_idx=1, device=device)
    pos_emb = model.encode_docs(positives, batch_size=batch_size, device=device)
    flat_negatives = [text for row in negatives for text in row]
    neg_emb = model.encode_docs(flat_negatives, batch_size=batch_size * num_negatives, device=device)
    return [q_emb], pos_emb, neg_emb.view(batch_size, num_negatives, -1)


def run_sanity_checks(
    model: LoopMatryoshkaRetriever,
    batch: Dict[str, Any],
    q_loops: list[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    loss_dict: Dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> None:
    batch_size = len(batch["queries"])
    num_negatives = len(batch["negatives"][0])
    expected_loops = 1 if get_version_spec(args.version).is_standard_family else int(args.tmax)
    if len(q_loops) != expected_loops:
        raise AssertionError(f"Expected {expected_loops} query loop tensors, got {len(q_loops)}.")
    for idx, q_emb in enumerate(q_loops, start=1):
        expected_shape = (batch_size, model.embedding_dim)
        if tuple(q_emb.shape) != expected_shape:
            raise AssertionError(f"h_{idx} shape mismatch: expected {expected_shape}, got {tuple(q_emb.shape)}.")
    if tuple(pos_emb.shape) != (batch_size, model.embedding_dim):
        raise AssertionError(f"pos_emb shape mismatch: got {tuple(pos_emb.shape)}.")
    if tuple(neg_emb.shape) != (batch_size, num_negatives, model.embedding_dim):
        raise AssertionError(f"neg_emb shape mismatch: got {tuple(neg_emb.shape)}.")

    loss_inbatch = loss_dict.get("loss_inbatch", loss_dict.get("loss_inbatch_avg"))
    if loss_inbatch is None or float(loss_inbatch.detach().cpu().item()) != 0.0:
        raise AssertionError("In-batch loss must be logged as 0.0 and excluded from total loss.")

    if get_version_spec(args.version).loss_type == "loopwise":
        loop_loss_values = [loss_dict[f"loss_t{idx}"] for idx in range(1, int(args.tmax) + 1)]
        expected_loss = torch.stack(loop_loss_values).mean()
    else:
        expected_loss = loss_dict.get("loss_hard", loss_dict.get("final_loop_loss"))
    if expected_loss is None or not torch.allclose(loss_dict["loss"].detach(), expected_loss.detach(), rtol=1e-4, atol=1e-5):
        raise AssertionError("Total loss must equal hard-negative loss only.")

    debug = model.last_query_loop_debug
    print("Sanity checks:")
    print(f"  version={args.version}")
    print(f"  loop_impl={model.loop_impl}")
    print(f"  use_inbatch={args.use_inbatch}")
    print(f"  tmax={args.tmax}")
    print(f"  detach_memory={model.detach_memory}")
    print(f"  q_loops={len(q_loops)} shapes={[tuple(t.shape) for t in q_loops]}")
    print(f"  pos_emb_shape={tuple(pos_emb.shape)} neg_emb_shape={tuple(neg_emb.shape)}")
    print(
        "  memory_tokens: "
        f"included={debug.get('memory_tokens_included', False)} "
        f"loop_memory_mode={debug.get('loop_memory_mode', 'mean_pool')} "
        f"query_len={debug.get('query_token_length')} "
        f"last_memory_tokens={debug.get('last_memory_tokens')} "
        f"last_input_len={debug.get('last_input_length')}"
    )
    print(f"  query_pooling_excludes_memory_tokens={debug.get('pooling_excludes_memory_tokens', True)}")
    print("  loss_check=hard_negative_only")


def log_row(
    args: argparse.Namespace,
    loss_dict: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    global_step: int,
    epoch_value: float,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "global_step": global_step,
        "epoch": epoch_value,
        "version": args.version,
        "total_loss": tensor_to_float(loss_dict["loss"]),
        "loss_hard": tensor_to_float(loss_dict.get("loss_hard", loss_dict.get("loss_hard_avg"))),
        "loss_inbatch": tensor_to_float(loss_dict.get("loss_inbatch", loss_dict.get("loss_inbatch_avg"))),
        "learning_rate_encoder": current_lr(optimizer, 0),
    }
    if torch.cuda.is_available():
        row["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
    loss_type = get_version_spec(args.version).loss_type
    if loss_type == "loopwise":
        for idx in range(1, args.tmax + 1):
            key = f"loss_t{idx}"
            if key in loss_dict:
                row[key] = tensor_to_float(loss_dict[key])
    if loss_type == "final_loop" and "final_loop_loss" in loss_dict:
        row["final_loop_loss"] = tensor_to_float(loss_dict["final_loop_loss"])
    return row


def save_checkpoint(
    model: LoopMatryoshkaRetriever,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    checkpoint_dir: Path,
) -> None:
    training_args = vars(args).copy()
    model.save_model(checkpoint_dir, training_args=training_args, optimizer=optimizer)


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    output_dir = ensure_dir(args.output_dir)
    log_path = output_dir / "train_log.jsonl"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.bf16 or args.fp16) and device.type != "cuda":
        print("Mixed precision requested, but CUDA is unavailable. Running in full precision on CPU.")
    autocast_enabled = device.type == "cuda" and (args.bf16 or args.fp16)
    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.fp16))

    dataset = RLHNRetrievalDataset(
        dataset_name=args.dataset_name,
        train_sample_size=args.train_sample_size,
        num_negatives=args.num_negatives,
        seed=args.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    model = LoopMatryoshkaRetriever(
        model_name_or_path=args.model_name_or_path,
        tmax=args.tmax,
        max_query_length=args.max_query_length,
        max_doc_length=args.max_doc_length,
        loop_impl=args.loop_impl,
        detach_memory=args.detach_memory,
        loop_memory_mode=args.loop_memory_mode,
    ).to(device)
    assert_encoder_only_trainable(model)

    optimizer = build_optimizer(model, args)
    updates_per_epoch = math.ceil(len(dataloader) / args.gradient_accumulation_steps)
    total_steps = int(args.epochs) * updates_per_epoch
    if args.max_steps is not None:
        total_steps = min(total_steps, int(args.max_steps))
    warmup_steps = int(total_steps * float(args.warmup_ratio))
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )

    print(f"Training {args.version} on {len(dataset)} samples for up to {total_steps} optimizer steps.")
    print(
        "Experiment config: "
        f"loop_impl={args.loop_impl}, use_inbatch={args.use_inbatch}, "
        f"tmax={args.tmax}, detach_memory={args.detach_memory}, "
        f"loop_memory_mode={args.loop_memory_mode}"
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    global_step = 0
    should_stop = False
    sanity_checked = False

    for epoch in range(int(args.epochs)):
        progress = tqdm(dataloader, desc=f"epoch {epoch + 1}/{args.epochs}", dynamic_ncols=True)
        for batch_idx, batch in enumerate(progress):
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype,
                enabled=autocast_enabled,
            ):
                q_loops, pos_emb, neg_emb = encode_training_batch(model, batch, args.version)
                loss_dict = select_loss(args.version, q_loops, pos_emb, neg_emb, args)
                if not sanity_checked:
                    run_sanity_checks(model, batch, q_loops, pos_emb, neg_emb, loss_dict, args)
                    sanity_checked = True
                loss = loss_dict["loss"] / int(args.gradient_accumulation_steps)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            is_update_step = (batch_idx + 1) % int(args.gradient_accumulation_steps) == 0 or (batch_idx + 1) == len(dataloader)
            if not is_update_step:
                continue

            if scaler.is_enabled():
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            epoch_value = epoch + (batch_idx + 1) / max(1, len(dataloader))
            progress.set_postfix({"step": global_step, "loss": f"{tensor_to_float(loss_dict['loss']):.4f}"})

            if global_step % int(args.log_steps) == 0:
                append_jsonl(log_path, log_row(args, loss_dict, optimizer, global_step, epoch_value))

            if int(args.save_steps) > 0 and global_step % int(args.save_steps) == 0:
                save_checkpoint(model, optimizer, args, output_dir / f"checkpoint-{global_step}")

            if args.max_steps is not None and global_step >= int(args.max_steps):
                should_stop = True
                break

        if should_stop:
            break

    final_dir = output_dir / "final"
    model.save_model(final_dir, training_args=vars(args).copy(), optimizer=None)
    print(f"Saved final model to {final_dir}")


if __name__ == "__main__":
    main()
