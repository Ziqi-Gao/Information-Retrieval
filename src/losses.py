from typing import Dict, List, Sequence

import torch
import torch.nn.functional as F


def retrieval_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = False,
    inbatch_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if inbatch_weight < 0:
        raise ValueError("inbatch_weight must be non-negative.")
    batch_size = q_emb.size(0)
    pos_scores = (q_emb * pos_emb).sum(dim=-1, keepdim=True)
    neg_scores = torch.einsum("bd,bkd->bk", q_emb, neg_emb)

    hard_logits = torch.cat([pos_scores, neg_scores], dim=1) / tau
    hard_targets = torch.zeros(batch_size, dtype=torch.long, device=q_emb.device)
    loss_hard = F.cross_entropy(hard_logits, hard_targets)
    loss_inbatch = torch.zeros((), dtype=loss_hard.dtype, device=q_emb.device)
    loss = loss_hard

    if use_inbatch:
        inbatch_logits = torch.matmul(q_emb, pos_emb.transpose(0, 1)) / tau
        inbatch_targets = torch.arange(batch_size, dtype=torch.long, device=q_emb.device)
        loss_inbatch = F.cross_entropy(inbatch_logits, inbatch_targets)
        loss = loss_hard + float(inbatch_weight) * loss_inbatch

    return {
        "loss": loss,
        "loss_hard": loss_hard.detach(),
        "loss_inbatch": loss_inbatch.detach(),
    }


def label_smoothed_retrieval_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    label_smoothing: float = 0.05,
) -> Dict[str, torch.Tensor]:
    if not 0.0 <= float(label_smoothing) < 1.0:
        raise ValueError("label_smoothing must be in [0, 1).")
    batch_size = q_emb.size(0)
    pos_scores = (q_emb * pos_emb).sum(dim=-1, keepdim=True)
    neg_scores = torch.einsum("bd,bkd->bk", q_emb, neg_emb)
    logits = torch.cat([pos_scores, neg_scores], dim=1) / tau
    num_classes = logits.size(1)
    if num_classes < 2:
        raise ValueError("label-smoothed retrieval loss requires at least one negative.")

    smoothing = float(label_smoothing)
    target = torch.full_like(logits, smoothing / float(num_classes - 1))
    target[:, 0] = 1.0 - smoothing
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(target * log_probs).sum(dim=1).mean()
    hard_targets = torch.zeros(batch_size, dtype=torch.long, device=q_emb.device)
    loss_hard = F.cross_entropy(logits, hard_targets)
    loss_inbatch = torch.zeros((), dtype=loss.dtype, device=q_emb.device)
    return {
        "loss": loss,
        "loss_hard": loss_hard.detach(),
        "loss_inbatch": loss_inbatch.detach(),
        "label_smoothing": torch.tensor(smoothing, dtype=loss.dtype, device=loss.device),
    }


def loopwise_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = False,
    inbatch_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    loop_losses = []
    hard_losses = []
    inbatch_losses = []
    output: Dict[str, torch.Tensor] = {}

    for idx, q_emb in enumerate(q_loops, start=1):
        loss_dict = retrieval_loss(
            q_emb,
            pos_emb,
            neg_emb,
            tau=tau,
            use_inbatch=use_inbatch,
            inbatch_weight=inbatch_weight,
        )
        loop_losses.append(loss_dict["loss"])
        hard_losses.append(loss_dict["loss_hard"])
        inbatch_losses.append(loss_dict["loss_inbatch"])
        output[f"loss_t{idx}"] = loss_dict["loss"].detach()

    output["loss"] = torch.stack(loop_losses).mean()
    output["loss_hard_avg"] = torch.stack(hard_losses).mean()
    output["loss_inbatch_avg"] = torch.stack(inbatch_losses).mean()
    return output


def loopwise_label_smoothed_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    label_smoothing: float = 0.05,
) -> Dict[str, torch.Tensor]:
    loop_losses = []
    hard_losses = []
    output: Dict[str, torch.Tensor] = {}

    for idx, q_emb in enumerate(q_loops, start=1):
        loss_dict = label_smoothed_retrieval_loss(
            q_emb,
            pos_emb,
            neg_emb,
            tau=tau,
            label_smoothing=label_smoothing,
        )
        loop_losses.append(loss_dict["loss"])
        hard_losses.append(loss_dict["loss_hard"])
        output[f"loss_t{idx}"] = loss_dict["loss"].detach()

    zero = torch.zeros((), dtype=loop_losses[0].dtype, device=loop_losses[0].device)
    output["loss"] = torch.stack(loop_losses).mean()
    output["loss_hard_avg"] = torch.stack(hard_losses).mean()
    output["loss_inbatch_avg"] = zero
    output["label_smoothing"] = torch.tensor(float(label_smoothing), dtype=loop_losses[0].dtype, device=loop_losses[0].device)
    return output


def loopwise_sparse_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    loop_indices: Sequence[int],
    tau: float = 0.05,
    use_inbatch: bool = False,
    inbatch_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if not loop_indices:
        raise ValueError("loop_indices must contain at least one loop index.")
    max_loop = len(q_loops)
    normalized_indices = sorted({int(idx) for idx in loop_indices})
    if normalized_indices[0] < 1 or normalized_indices[-1] > max_loop:
        raise ValueError(f"loop_indices must be in [1, {max_loop}], got {normalized_indices}.")

    loop_losses = []
    hard_losses = []
    inbatch_losses = []
    output: Dict[str, torch.Tensor] = {}

    for idx in normalized_indices:
        loss_dict = retrieval_loss(
            q_loops[idx - 1],
            pos_emb,
            neg_emb,
            tau=tau,
            use_inbatch=use_inbatch,
            inbatch_weight=inbatch_weight,
        )
        loop_losses.append(loss_dict["loss"])
        hard_losses.append(loss_dict["loss_hard"])
        inbatch_losses.append(loss_dict["loss_inbatch"])
        output[f"loss_t{idx}"] = loss_dict["loss"].detach()

    output["loss"] = torch.stack(loop_losses).mean()
    output["loss_hard_avg"] = torch.stack(hard_losses).mean()
    output["loss_inbatch_avg"] = torch.stack(inbatch_losses).mean()
    output["loop_loss_indices"] = torch.tensor(normalized_indices, dtype=torch.long, device=output["loss"].device)
    return output


def loopwise_tail_weighted_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = False,
    inbatch_weight: float = 1.0,
    gamma: float = 1.25,
) -> Dict[str, torch.Tensor]:
    if gamma <= 0:
        raise ValueError("gamma must be positive.")

    loop_losses = []
    hard_losses = []
    inbatch_losses = []
    output: Dict[str, torch.Tensor] = {}

    for idx, q_emb in enumerate(q_loops, start=1):
        loss_dict = retrieval_loss(
            q_emb,
            pos_emb,
            neg_emb,
            tau=tau,
            use_inbatch=use_inbatch,
            inbatch_weight=inbatch_weight,
        )
        loop_losses.append(loss_dict["loss"])
        hard_losses.append(loss_dict["loss_hard"])
        inbatch_losses.append(loss_dict["loss_inbatch"])
        output[f"loss_t{idx}"] = loss_dict["loss"].detach()

    weights = torch.tensor(
        [float(gamma) ** idx for idx in range(len(loop_losses))],
        dtype=loop_losses[0].dtype,
        device=loop_losses[0].device,
    )
    weights = weights / weights.sum().clamp(min=1e-12)
    stacked_losses = torch.stack(loop_losses)
    output["loss"] = (stacked_losses * weights).sum()
    output["loss_hard_avg"] = torch.stack(hard_losses).mean()
    output["loss_inbatch_avg"] = torch.stack(inbatch_losses).mean()
    output["loop_loss_gamma"] = torch.tensor(float(gamma), dtype=stacked_losses.dtype, device=stacked_losses.device)
    return output


def loopwise_consistency_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = False,
    inbatch_weight: float = 1.0,
    consistency_lambda: float = 0.05,
) -> Dict[str, torch.Tensor]:
    if consistency_lambda < 0:
        raise ValueError("consistency_lambda must be non-negative.")
    output = loopwise_loss(
        q_loops,
        pos_emb,
        neg_emb,
        tau=tau,
        use_inbatch=use_inbatch,
        inbatch_weight=inbatch_weight,
    )
    if len(q_loops) < 2 or consistency_lambda == 0:
        consistency = torch.zeros((), dtype=output["loss"].dtype, device=output["loss"].device)
    else:
        penalties = []
        for prev, current in zip(q_loops[:-1], q_loops[1:]):
            penalties.append(1.0 - (prev * current).sum(dim=-1).mean())
        consistency = torch.stack(penalties).mean()
    output["loop_consistency_loss"] = consistency.detach()
    output["loss_loopwise_base"] = output["loss"].detach()
    output["loss"] = output["loss"] + float(consistency_lambda) * consistency
    return output


def pairwise_ranking_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    pos_scores = (q_emb * pos_emb).sum(dim=-1, keepdim=True)
    neg_scores = torch.einsum("bd,bkd->bk", q_emb, neg_emb)
    pairwise_terms = F.softplus((neg_scores - pos_scores + float(margin)) / tau)
    loss_pairwise = pairwise_terms.mean()
    loss_inbatch = torch.zeros((), dtype=loss_pairwise.dtype, device=q_emb.device)
    return {
        "loss": loss_pairwise,
        "loss_pairwise": loss_pairwise.detach(),
        "loss_hard": loss_pairwise.detach(),
        "loss_inbatch": loss_inbatch.detach(),
    }


def loopwise_pairwise_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    loop_losses = []
    pairwise_losses = []
    output: Dict[str, torch.Tensor] = {}

    for idx, q_emb in enumerate(q_loops, start=1):
        loss_dict = pairwise_ranking_loss(q_emb, pos_emb, neg_emb, tau=tau, margin=margin)
        loop_losses.append(loss_dict["loss"])
        pairwise_losses.append(loss_dict["loss_pairwise"])
        output[f"loss_t{idx}"] = loss_dict["loss"].detach()

    zero = torch.zeros((), dtype=loop_losses[0].dtype, device=loop_losses[0].device)
    output["loss"] = torch.stack(loop_losses).mean()
    output["loss_pairwise_avg"] = torch.stack(pairwise_losses).mean()
    output["loss_hard_avg"] = output["loss_pairwise_avg"]
    output["loss_inbatch_avg"] = zero
    return output


def final_loop_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = False,
) -> Dict[str, torch.Tensor]:
    output = retrieval_loss(q_loops[-1], pos_emb, neg_emb, tau=tau, use_inbatch=use_inbatch)
    output["final_loop_loss"] = output["loss"].detach()
    return output


def standard_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = False,
) -> Dict[str, torch.Tensor]:
    return retrieval_loss(q_emb, pos_emb, neg_emb, tau=tau, use_inbatch=use_inbatch)
