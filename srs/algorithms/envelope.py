"""Serializable result objects shared by all algorithms."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
import uuid

SCHEMA_VERSION = "1.0"


@dataclass
class Evidence:
    """One element's signed contribution to a match."""

    element: str
    contribution: float
    imputed: bool = False

    def to_dict(self):
        return {
            "element": self.element,
            "contribution": float(self.contribution),
            "imputed": bool(self.imputed),
        }


@dataclass
class Match:
    """One ranked reference result and its optional evidence."""

    rank: int
    similarity: float
    reference_sample_id: Optional[int] = None
    deposit_id: str = ""
    deposit_name: str = ""
    deposit_class: str = ""
    # Extra metrics keep their own names beside the normalised score.
    scores: dict = field(default_factory=dict)
    confidence: Optional[dict] = None
    supporting: list = field(default_factory=list)
    conflicting: list = field(default_factory=list)

    def to_dict(self):
        match = {
            "rank": self.rank,
            "reference_sample_id": self.reference_sample_id,
            "deposit_id": self.deposit_id,
            "deposit_name": self.deposit_name,
            "deposit_class": self.deposit_class,
            # Exact matches from some methods are integers; the API uses floats.
            "scores": {"similarity": float(self.similarity), **self.scores},
        }

        if self.confidence:
            match["confidence"] = self.confidence

        if self.supporting or self.conflicting:
            match["evidence"] = {
                "supporting": [item.to_dict() for item in self.supporting],
                "conflicting": [item.to_dict() for item in self.conflicting],
            }

        return match


@dataclass
class RunResult:
    """Results and settings recorded for one algorithm run."""

    algorithm_id: str
    algorithm_version: str
    matches: list
    runtime_ms: float
    algorithm_params: dict = field(default_factory=dict)
    preprocessing: dict = field(default_factory=dict)
    reference_library_version: str = ""
    dataset_id: Optional[Any] = None
    sample_results: Optional[list] = None
    projection: Optional[dict] = None
    warnings: list = field(default_factory=list)
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_envelope(self):
        envelope = {
            "schema_version": SCHEMA_VERSION,
            "run": {
                "run_id": self.run_id,
                "created_at": self.created_at,
                "dataset_id": self.dataset_id,
                "algorithm": {
                    "id": self.algorithm_id,
                    "version": self.algorithm_version,
                    "params": self.algorithm_params,
                },
                "preprocessing": self.preprocessing,
                "reference_library_version": self.reference_library_version,
                "runtime_ms": self.runtime_ms,
            },
            "matches": [match.to_dict() for match in self.matches],
            "warnings": list(self.warnings),
        }

        # Leave unsupported result sections out instead of returning null.
        if self.sample_results is not None:
            envelope["sample_results"] = self.sample_results

        if self.projection is not None:
            envelope["projection"] = self.projection

        return envelope


def validate_envelope(envelope):
    """Return result-envelope validation errors, or an empty list when valid."""
    problems = []

    if envelope.get("schema_version") != SCHEMA_VERSION:
        problems.append(
            f"schema_version must be {SCHEMA_VERSION!r}, "
            f"got {envelope.get('schema_version')!r}"
        )

    run = envelope.get("run")
    if not isinstance(run, dict):
        problems.append("run block is missing")
    else:
        for key in ("run_id", "created_at", "algorithm", "preprocessing", "runtime_ms"):
            if key not in run:
                problems.append(f"run.{key} is missing")

        algorithm = run.get("algorithm")
        if isinstance(algorithm, dict):
            for key in ("id", "version", "params"):
                if key not in algorithm:
                    problems.append(f"run.algorithm.{key} is missing")
        else:
            problems.append("run.algorithm block is missing")

        if not isinstance(run.get("runtime_ms"), (int, float)):
            problems.append("run.runtime_ms must be a number")

    matches = envelope.get("matches")
    if not isinstance(matches, list):
        problems.append("matches must be a list")
        return problems

    for position, match in enumerate(matches):
        scores = match.get("scores")
        if not isinstance(scores, dict) or "similarity" not in scores:
            problems.append(f"matches[{position}].scores.similarity is missing")
            continue

        similarity = scores["similarity"]
        if not isinstance(similarity, float):
            problems.append(
                f"matches[{position}].scores.similarity must be a float, "
                f"got {type(similarity).__name__}"
            )
        elif not 0.0 <= similarity <= 1.0:
            problems.append(
                f"matches[{position}].scores.similarity must be within [0, 1], "
                f"got {similarity}"
            )

        if "rank" not in match:
            problems.append(f"matches[{position}].rank is missing")

        evidence = match.get("evidence")
        if evidence is not None:
            for side in ("supporting", "conflicting"):
                for entry in evidence.get(side, []):
                    missing = {"element", "contribution", "imputed"} - set(entry)
                    if missing:
                        problems.append(
                            f"matches[{position}].evidence.{side} entry is missing "
                            f"{sorted(missing)}"
                        )

    return problems
