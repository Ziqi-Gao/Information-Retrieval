from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class VersionSpec:
    name: str
    family: str
    loss_type: str
    description: str
    eval_all_loops: bool
    plot_kind: str
    color: str
    loop_memory_mode: Optional[str] = None
    loop_query_mode: Optional[str] = None
    embedding_pooling_mode: Optional[str] = None
    passage_sampling_strategy: Optional[str] = None

    @property
    def is_standard_family(self) -> bool:
        return self.family == "standard"

    @property
    def is_loop_family(self) -> bool:
        return self.family == "loop"

    @property
    def is_baseline(self) -> bool:
        return self.plot_kind == "baseline"

    @property
    def is_curve(self) -> bool:
        return self.plot_kind == "curve"


VERSION_SPECS: Dict[str, VersionSpec] = {
    "standard": VersionSpec(
        name="standard",
        family="standard",
        loss_type="standard",
        description="No-loop hard-negative retriever.",
        eval_all_loops=False,
        plot_kind="baseline",
        color="black",
    ),
    "standard_role_prompt": VersionSpec(
        name="standard_role_prompt",
        family="standard",
        loss_type="standard",
        description="No-loop hard-negative retriever trained and evaluated with query/document role prefixes.",
        eval_all_loops=False,
        plot_kind="curve",
        color="#4daf4a",
    ),
    "standard_dim_mrl": VersionSpec(
        name="standard_dim_mrl",
        family="standard",
        loss_type="standard_dim_mrl",
        description="No-loop hard-negative retriever with dimensional Matryoshka embedding supervision.",
        eval_all_loops=False,
        plot_kind="curve",
        color="#377eb8",
    ),
    "standard_role_prompt_dim_mrl": VersionSpec(
        name="standard_role_prompt_dim_mrl",
        family="standard",
        loss_type="standard_dim_mrl",
        description="No-loop retriever combining role prefixes with dimensional Matryoshka supervision.",
        eval_all_loops=False,
        plot_kind="curve",
        color="#984ea3",
    ),
    "standard_seeded_sampling": VersionSpec(
        name="standard_seeded_sampling",
        family="standard",
        loss_type="standard",
        description="No-loop hard-negative retriever trained with seeded random positive and negative passage sampling.",
        eval_all_loops=False,
        plot_kind="curve",
        color="#a65628",
        passage_sampling_strategy="seeded_random",
    ),
    "standard_inbatch_hybrid": VersionSpec(
        name="standard_inbatch_hybrid",
        family="standard",
        loss_type="standard_inbatch_hybrid",
        description="No-loop hard-negative retriever with candidate-only in-batch positive classification.",
        eval_all_loops=False,
        plot_kind="curve",
        color="#ff7f00",
    ),
    "loop_final": VersionSpec(
        name="loop_final",
        family="loop",
        loss_type="final_loop",
        description="Parameter-free memory loop retriever supervised only at the final loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#1f77b4",
    ),
    "loop_matryoshka": VersionSpec(
        name="loop_matryoshka",
        family="loop",
        loss_type="loopwise",
        description="Parameter-free memory loop retriever supervised at every loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#d62728",
    ),
    "loop_matryoshka_first_token": VersionSpec(
        name="loop_matryoshka_first_token",
        family="loop",
        loss_type="loopwise",
        description="First-token memory loop retriever supervised at every loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#8c564b",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_final_first_token": VersionSpec(
        name="loop_final_first_token",
        family="loop",
        loss_type="final_loop",
        description="First-token memory loop retriever supervised only at the final loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#bcbd22",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_matryoshka_token_concat": VersionSpec(
        name="loop_matryoshka_token_concat",
        family="loop",
        loss_type="loopwise",
        description="Token-concat memory loop retriever supervised at every loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#e377c2",
        loop_memory_mode="token_concat",
        loop_query_mode="initial_embedding",
    ),
    "loop_tail_weighted_first_token": VersionSpec(
        name="loop_tail_weighted_first_token",
        family="loop",
        loss_type="loopwise_tail_weighted",
        description="First-token memory loop retriever with deeper loops weighted more heavily during loopwise training.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#7f7f7f",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_consistency_first_token": VersionSpec(
        name="loop_consistency_first_token",
        family="loop",
        loss_type="loopwise_consistency",
        description="First-token memory loop retriever with adjacent-loop query embedding consistency regularization.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#aec7e8",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_inbatch_hybrid_first_token": VersionSpec(
        name="loop_inbatch_hybrid_first_token",
        family="loop",
        loss_type="loopwise_inbatch_hybrid",
        description="First-token memory loop retriever with hard negatives plus candidate-only in-batch positives.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#393b79",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_pairwise_first_token": VersionSpec(
        name="loop_pairwise_first_token",
        family="loop",
        loss_type="loopwise_pairwise",
        description="First-token memory loop retriever trained with loopwise pairwise softplus ranking loss.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#637939",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_matryoshka_first_token_seeded_sampling": VersionSpec(
        name="loop_matryoshka_first_token_seeded_sampling",
        family="loop",
        loss_type="loopwise",
        description="First-token memory loop retriever trained with seeded random positive and negative passage sampling.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#8dd3c7",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
        passage_sampling_strategy="seeded_random",
    ),
    "loop_matryoshka_first_token_first_pool": VersionSpec(
        name="loop_matryoshka_first_token_first_pool",
        family="loop",
        loss_type="loopwise",
        description="First-token memory loop retriever using the first query/doc token as the retrieval embedding.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#fb8072",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
        embedding_pooling_mode="first_token",
    ),
    "loop_two_stage_first_token": VersionSpec(
        name="loop_two_stage_first_token",
        family="loop",
        loss_type="two_stage_loopwise",
        description="First-token memory loop retriever with standard hard-negative warmup before loopwise training.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#80b1d3",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_matryoshka_first_token_middle_negatives": VersionSpec(
        name="loop_matryoshka_first_token_middle_negatives",
        family="loop",
        loss_type="loopwise",
        description="First-token memory loop retriever trained with a deterministic middle negative window.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#fdb462",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
        passage_sampling_strategy="middle_negatives",
    ),
    "loop_sparse_first_token": VersionSpec(
        name="loop_sparse_first_token",
        family="loop",
        loss_type="loopwise_sparse",
        description="First-token memory loop retriever supervised only at a predeclared sparse set of loop depths.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#b3de69",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_label_smooth_first_token": VersionSpec(
        name="loop_label_smooth_first_token",
        family="loop",
        loss_type="loopwise_label_smoothed",
        description="First-token memory loop retriever trained with label-smoothed listwise hard-negative loss.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#fccde5",
        loop_memory_mode="first_token",
        loop_query_mode="initial_embedding",
    ),
    "loop_final_qdoc_mean_pool": VersionSpec(
        name="loop_final_qdoc_mean_pool",
        family="loop",
        loss_type="final_loop_qdoc",
        description="Mean-pool memory loop retriever trained with looped query and looped document embeddings at the final loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#66c2a5",
        loop_memory_mode="mean_pool",
        loop_query_mode="initial_embedding",
    ),
    "loop_final_recurrent_mean_pool": VersionSpec(
        name="loop_final_recurrent_mean_pool",
        family="loop",
        loss_type="final_loop",
        description="Mean-pool memory loop retriever with recurrent query hidden states supervised only at the final loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#2ca02c",
        loop_memory_mode="mean_pool",
        loop_query_mode="recurrent_hidden",
    ),
    "loop_matryoshka_recurrent_mean_pool": VersionSpec(
        name="loop_matryoshka_recurrent_mean_pool",
        family="loop",
        loss_type="loopwise",
        description="Mean-pool memory loop retriever with recurrent query hidden states supervised at every loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#9467bd",
        loop_memory_mode="mean_pool",
        loop_query_mode="recurrent_hidden",
    ),
    "loop_final_recurrent_no_memory": VersionSpec(
        name="loop_final_recurrent_no_memory",
        family="loop",
        loss_type="final_loop",
        description="No-memory recurrent query hidden-state retriever supervised only at the final loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#ff7f0e",
        loop_memory_mode="none",
        loop_query_mode="recurrent_hidden",
    ),
    "loop_matryoshka_recurrent_no_memory": VersionSpec(
        name="loop_matryoshka_recurrent_no_memory",
        family="loop",
        loss_type="loopwise",
        description="No-memory recurrent query hidden-state retriever supervised at every loop.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#17becf",
        loop_memory_mode="none",
        loop_query_mode="recurrent_hidden",
    ),
}

MAIN_VERSIONS: Sequence[str] = (
    "standard",
    "standard_role_prompt",
    "standard_dim_mrl",
    "standard_role_prompt_dim_mrl",
    "standard_seeded_sampling",
    "standard_inbatch_hybrid",
    "loop_final",
    "loop_matryoshka",
    "loop_matryoshka_first_token",
    "loop_final_first_token",
    "loop_matryoshka_token_concat",
    "loop_tail_weighted_first_token",
    "loop_consistency_first_token",
    "loop_inbatch_hybrid_first_token",
    "loop_pairwise_first_token",
    "loop_matryoshka_first_token_seeded_sampling",
    "loop_matryoshka_first_token_first_pool",
    "loop_two_stage_first_token",
    "loop_matryoshka_first_token_middle_negatives",
    "loop_sparse_first_token",
    "loop_label_smooth_first_token",
    "loop_final_qdoc_mean_pool",
    "loop_final_recurrent_mean_pool",
    "loop_matryoshka_recurrent_mean_pool",
    "loop_final_recurrent_no_memory",
    "loop_matryoshka_recurrent_no_memory",
)


def get_version_spec(version: str) -> VersionSpec:
    try:
        return VERSION_SPECS[version]
    except KeyError as exc:
        known = ", ".join(version_names())
        raise ValueError(f"Unknown experiment version {version!r}. Known versions: {known}") from exc


def version_names(include_extra_baselines: bool = True) -> List[str]:
    if include_extra_baselines:
        return sorted(VERSION_SPECS)
    return list(MAIN_VERSIONS)


def standard_versions() -> List[str]:
    return [name for name, spec in VERSION_SPECS.items() if spec.is_standard_family]


def loop_versions() -> List[str]:
    return [name for name, spec in VERSION_SPECS.items() if spec.is_loop_family]


def baseline_versions() -> List[str]:
    return [name for name, spec in VERSION_SPECS.items() if spec.is_baseline]


def curve_versions() -> List[str]:
    return [name for name, spec in VERSION_SPECS.items() if spec.is_curve]


def versions_for_plot(rows_versions: Iterable[str]) -> List[str]:
    present = set(rows_versions)
    return [name for name in version_names() if name in present]
