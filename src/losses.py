from typing import Dict, List

import torch
import torch.nn.functional as F


def retrieval_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = True,
) -> Dict[str, torch.Tensor]:
    del use_inbatch
    batch_size = q_emb.size(0)
    pos_scores = (q_emb * pos_emb).sum(dim=-1, keepdim=True)
    neg_scores = torch.einsum("bd,bkd->bk", q_emb, neg_emb)

    hard_logits = torch.cat([pos_scores, neg_scores], dim=1) / tau
    hard_targets = torch.zeros(batch_size, dtype=torch.long, device=q_emb.device)
    loss_hard = F.cross_entropy(hard_logits, hard_targets)
    loss_inbatch = torch.zeros((), dtype=loss_hard.dtype, device=q_emb.device)

    return {
        "loss": loss_hard,
        "loss_hard": loss_hard.detach(),
        "loss_inbatch": loss_inbatch.detach(),
    }


def loopwise_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = True,
) -> Dict[str, torch.Tensor]:
    loop_losses = []
    hard_losses = []
    inbatch_losses = []
    output: Dict[str, torch.Tensor] = {}

    for idx, q_emb in enumerate(q_loops, start=1):
        loss_dict = retrieval_loss(q_emb, pos_emb, neg_emb, tau=tau, use_inbatch=use_inbatch)
        loop_losses.append(loss_dict["loss"])
        hard_losses.append(loss_dict["loss_hard"])
        inbatch_losses.append(loss_dict["loss_inbatch"])
        output[f"loss_t{idx}"] = loss_dict["loss"].detach()

    output["loss"] = torch.stack(loop_losses).mean()
    output["loss_hard_avg"] = torch.stack(hard_losses).mean()
    output["loss_inbatch_avg"] = torch.stack(inbatch_losses).mean()
    return output


def final_loop_loss(
    q_loops: List[torch.Tensor],
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = True,
) -> Dict[str, torch.Tensor]:
    output = retrieval_loss(q_loops[-1], pos_emb, neg_emb, tau=tau, use_inbatch=use_inbatch)
    output["final_loop_loss"] = output["loss"].detach()
    return output


def standard_loss(
    q_emb: torch.Tensor,
    pos_emb: torch.Tensor,
    neg_emb: torch.Tensor,
    tau: float = 0.05,
    use_inbatch: bool = True,
) -> Dict[str, torch.Tensor]:
    return retrieval_loss(q_emb, pos_emb, neg_emb, tau=tau, use_inbatch=use_inbatch)
