from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class VersionSpec:
    name: str
    family: str
    loss_type: str
    description: str
    eval_all_loops: bool
    plot_kind: str
    color: str
    use_memory_history: bool = True
    memory_history_mode: str = "full"

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
        use_memory_history=False,
        memory_history_mode="none",
    ),
    "standard_more_steps": VersionSpec(
        name="standard_more_steps",
        family="standard",
        loss_type="standard",
        description="No-loop hard-negative retriever trained for more optimizer steps.",
        eval_all_loops=False,
        plot_kind="baseline",
        color="#2ca02c",
        use_memory_history=False,
        memory_history_mode="none",
    ),
    "loop_final": VersionSpec(
        name="loop_final",
        family="loop",
        loss_type="final_loop",
        description="Memory-token loop retriever supervised only at the final loop, using full memory history.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#1f77b4",
        memory_history_mode="full",
    ),
    "loop_final_last": VersionSpec(
        name="loop_final_last",
        family="loop",
        loss_type="final_loop",
        description="Loop retriever supervised only at the final loop, using only the previous loop state as memory.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#9467bd",
        memory_history_mode="last",
    ),
    "loop_final_none": VersionSpec(
        name="loop_final_none",
        family="loop",
        loss_type="final_loop",
        description="Loop retriever supervised only at the final loop, without recurrent memory-state feedback.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#8c564b",
        use_memory_history=False,
        memory_history_mode="none",
    ),
    "loop_matryoshka": VersionSpec(
        name="loop_matryoshka",
        family="loop",
        loss_type="loopwise",
        description="Memory-token loop retriever supervised at every loop, using full memory history.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#d62728",
        memory_history_mode="full",
    ),
    "loop_matryoshka_last": VersionSpec(
        name="loop_matryoshka_last",
        family="loop",
        loss_type="loopwise",
        description="Loop retriever supervised at every loop, using only the previous loop state as memory.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#ff7f0e",
        memory_history_mode="last",
    ),
    "loop_matryoshka_none": VersionSpec(
        name="loop_matryoshka_none",
        family="loop",
        loss_type="loopwise",
        description="Loop retriever supervised at every loop, without recurrent memory-state feedback.",
        eval_all_loops=True,
        plot_kind="curve",
        color="#17becf",
        use_memory_history=False,
        memory_history_mode="none",
    ),
}

MAIN_VERSIONS: Sequence[str] = ("standard", "loop_final", "loop_matryoshka")


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
