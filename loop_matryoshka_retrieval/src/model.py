import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .utils import ensure_dir, write_json


class LoopMatryoshkaRetriever(nn.Module):
    def __init__(
        self,
        model_name_or_path: str,
        tmax: int = 10,
        max_query_length: int = 128,
        max_doc_length: int = 512,
        loop_impl: str = "memory_token",
        detach_memory: bool = False,
        use_memory_history: bool = True,
        use_projection: bool = True,
        projection_dim: Optional[int] = None,
        dropout: float = 0.1,
        loop_alpha_init: float = 0.1,
    ) -> None:
        super().__init__()
        if tmax < 1:
            raise ValueError("tmax must be >= 1.")

        self.model_name_or_path = model_name_or_path
        self.tmax = int(tmax)
        self.max_query_length = int(max_query_length)
        self.max_doc_length = int(max_doc_length)
        self.loop_impl = str(loop_impl)
        if self.loop_impl != "memory_token":
            raise ValueError(f"Only loop_impl='memory_token' is supported, got {loop_impl!r}.")
        self.detach_memory = bool(detach_memory)
        self.use_memory_history = bool(use_memory_history)
        self.use_projection = bool(use_projection)
        self.requested_projection_dim = projection_dim
        self.dropout = float(dropout)
        self.loop_alpha_init = float(loop_alpha_init)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.encoder = AutoModel.from_pretrained(model_name_or_path)
        hidden_size = int(self.encoder.config.hidden_size)
        self.hidden_size = hidden_size
        self.embedding_dim = int(projection_dim) if self.use_projection and projection_dim is not None else hidden_size

        if self.use_projection:
            self.projection = nn.Linear(hidden_size, self.embedding_dim)
        else:
            self.projection = nn.Identity()

        self.memory_projection = nn.Linear(self.embedding_dim, hidden_size)
        self.memory_state_embeddings = nn.Embedding(self.tmax + 1, hidden_size)
        self.last_query_loop_debug: Dict[str, Any] = {}

    @staticmethod
    def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        return pooled

    def _device(self, device: Optional[Union[torch.device, str]] = None) -> torch.device:
        if device is not None:
            return torch.device(device)
        return next(self.parameters()).device

    def _encode_texts_once(self, texts: List[str], max_length: int, device: torch.device) -> torch.Tensor:
        if not texts:
            return torch.empty(0, self.embedding_dim, device=device)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        outputs = self.encoder(**encoded)
        pooled = self.mean_pool(outputs.last_hidden_state, encoded["attention_mask"])
        projected = self.projection(pooled)
        return F.normalize(projected, p=2, dim=-1)

    def encode_texts(
        self,
        texts: List[str],
        max_length: int,
        batch_size: int,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        device = self._device(device)
        if not texts:
            return torch.empty(0, self.embedding_dim, device=device)

        outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), int(batch_size)):
            outputs.append(self._encode_texts_once(texts[start : start + int(batch_size)], max_length, device))
        return torch.cat(outputs, dim=0)

    def _encode_query_loop_chunk(
        self,
        texts: List[str],
        device: torch.device,
        loop_limit: int,
    ) -> List[torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_query_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        h1 = F.normalize(self.projection(pooled), p=2, dim=-1)
        states = [h1]

        query_token_embeds = self.encoder.get_input_embeddings()(input_ids)
        batch_size = input_ids.size(0)
        query_token_length = input_ids.size(1)
        last_input_length = query_token_length
        last_memory_tokens = 0

        for t in range(2, loop_limit + 1):
            if self.use_memory_history:
                memory_tokens = []
                for idx, state in enumerate(states, start=1):
                    h_mem = state.detach() if self.detach_memory else state
                    state_ids = torch.full((batch_size,), idx, dtype=torch.long, device=device)
                    memory_tokens.append(self.memory_projection(h_mem) + self.memory_state_embeddings(state_ids))
                memory_tokens_tensor = torch.stack(memory_tokens, dim=1)

                memory_mask = torch.ones(
                    batch_size,
                    memory_tokens_tensor.size(1),
                    dtype=attention_mask.dtype,
                    device=device,
                )
                inputs_embeds = torch.cat([memory_tokens_tensor, query_token_embeds], dim=1)
                extended_attention_mask = torch.cat([memory_mask, attention_mask], dim=1)
                outputs = self.encoder(inputs_embeds=inputs_embeds, attention_mask=extended_attention_mask)
                query_hidden = outputs.last_hidden_state[:, memory_tokens_tensor.size(1) :, :]
                last_memory_tokens = memory_tokens_tensor.size(1)
                last_input_length = inputs_embeds.size(1)
            else:
                outputs = self.encoder(inputs_embeds=query_token_embeds, attention_mask=attention_mask)
                query_hidden = outputs.last_hidden_state
                last_memory_tokens = 0
                last_input_length = query_token_length
            pooled = self.mean_pool(query_hidden, attention_mask)
            ht = F.normalize(self.projection(pooled), p=2, dim=-1)
            states.append(ht)

        self.last_query_loop_debug = {
            "loop_impl": self.loop_impl,
            "use_memory_history": self.use_memory_history,
            "query_token_length": int(query_token_length),
            "last_memory_tokens": int(last_memory_tokens),
            "last_input_length": int(last_input_length),
            "memory_tokens_included": bool(last_memory_tokens > 0),
            "pooling_excludes_memory_tokens": True,
        }
        return states

    def encode_query_loops(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        return_all_loops: bool = True,
        device: Optional[Union[torch.device, str]] = None,
    ) -> Union[List[torch.Tensor], torch.Tensor]:
        device = self._device(device)
        if not texts:
            if return_all_loops:
                return [torch.empty(0, self.embedding_dim, device=device) for _ in range(self.tmax)]
            return torch.empty(0, self.embedding_dim, device=device)

        chunk_size = int(batch_size or len(texts))
        chunk_outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), chunk_size):
            states = self._encode_query_loop_chunk(texts[start : start + chunk_size], device, self.tmax)
            chunk_outputs.append(torch.stack(states, dim=0))
        stacked = torch.cat(chunk_outputs, dim=1)
        if return_all_loops:
            return list(stacked.unbind(dim=0))
        return stacked[-1]

    def encode_docs(
        self,
        texts: List[str],
        batch_size: int = 32,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        return self.encode_texts(texts, self.max_doc_length, batch_size, device)

    def encode_queries(
        self,
        texts: List[str],
        batch_size: int = 32,
        loop_idx: Optional[int] = None,
        return_all_loops: bool = False,
        device: Optional[Union[torch.device, str]] = None,
    ) -> torch.Tensor:
        device = self._device(device)
        if loop_idx is not None and not 1 <= int(loop_idx) <= self.tmax:
            raise ValueError(f"loop_idx must be in [1, {self.tmax}], got {loop_idx}.")
        target_loop = self.tmax if loop_idx is None else int(loop_idx)
        loop_limit = self.tmax if return_all_loops else target_loop

        chunk_outputs: List[torch.Tensor] = []
        for start in range(0, len(texts), int(batch_size)):
            states = self._encode_query_loop_chunk(texts[start : start + int(batch_size)], device, loop_limit)
            if return_all_loops:
                chunk_outputs.append(torch.stack(states, dim=0))
            else:
                chunk_outputs.append(states[target_loop - 1])

        if not chunk_outputs:
            if return_all_loops:
                return torch.empty(self.tmax, 0, self.embedding_dim, device=device)
            return torch.empty(0, self.embedding_dim, device=device)

        if return_all_loops:
            return torch.cat(chunk_outputs, dim=1)
        return torch.cat(chunk_outputs, dim=0)

    def forward_batch(
        self,
        queries: List[str],
        positives: List[str],
        negatives: List[List[str]],
    ) -> tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
        if len(queries) != len(positives) or len(queries) != len(negatives):
            raise ValueError("queries, positives, and negatives must have the same batch size.")
        if not negatives:
            raise ValueError("negatives must be non-empty.")

        batch_size = len(queries)
        num_negatives = len(negatives[0])
        if any(len(row) != num_negatives for row in negatives):
            raise ValueError("Every sample must have the same number of negatives.")

        q_stack = self.encode_queries(
            queries,
            batch_size=batch_size,
            return_all_loops=True,
            device=self._device(),
        )
        q_loops = list(q_stack.unbind(dim=0))
        pos_emb = self.encode_docs(positives, batch_size=batch_size, device=self._device())

        flat_negatives = [text for row in negatives for text in row]
        neg_emb = self.encode_docs(flat_negatives, batch_size=batch_size * num_negatives, device=self._device())
        neg_emb = neg_emb.view(batch_size, num_negatives, -1)
        return q_loops, pos_emb, neg_emb

    def loop_config(self) -> Dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "tmax": self.tmax,
            "max_query_length": self.max_query_length,
            "max_doc_length": self.max_doc_length,
            "loop_impl": self.loop_impl,
            "detach_memory": self.detach_memory,
            "use_memory_history": self.use_memory_history,
            "use_projection": self.use_projection,
            "projection_dim": self.embedding_dim,
            "dropout": self.dropout,
            "loop_alpha_init": self.loop_alpha_init,
        }

    def save_model(
        self,
        output_dir: Union[str, Path],
        training_args: Optional[Dict[str, Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ) -> None:
        output_dir = ensure_dir(output_dir)
        self.encoder.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        write_json(output_dir / "loop_config.json", self.loop_config())
        torch.save(self.state_dict(), output_dir / "model_state.pt")
        if training_args is not None:
            write_json(output_dir / "training_args.json", training_args)
        if optimizer is not None:
            torch.save(optimizer.state_dict(), output_dir / "optimizer.pt")


def load_model(
    checkpoint_dir: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
) -> LoopMatryoshkaRetriever:
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "loop_config.json"
    state_path = checkpoint_dir / "model_state.pt"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing loop_config.json in checkpoint directory: {checkpoint_dir}")
    if not state_path.exists():
        raise FileNotFoundError(f"Missing model_state.pt in checkpoint directory: {checkpoint_dir}")

    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["model_name_or_path"] = str(checkpoint_dir)
    model = LoopMatryoshkaRetriever(**config)
    state = torch.load(state_path, map_location=map_location or "cpu")
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(
            "Loaded checkpoint with non-strict state dict. "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    return model
