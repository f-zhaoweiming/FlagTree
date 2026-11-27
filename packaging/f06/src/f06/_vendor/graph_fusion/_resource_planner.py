"""Deterministic, resource-aware contiguous graph partition planning.

This module is pure scheduling metadata.  It does not generate code, invoke
kernels, capture timings, autotune, or certify a deployed plan.

The parent supplies an ordered sequence of compute-node IDs and profiled
observations for positional half-open intervals ``[start, end)``.  Observations
are usable only when they match the caller's exact context fingerprint and pass
all validation gates.  Planning is an exact dynamic program over contiguous
intervals; no unmeasured interval costs are inferred.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, TypeAlias

__all__ = [
    "DEPLOYMENT_ADVISORY",
    "NoPlan",
    "PartitionDecision",
    "PartitionPlan",
    "PlannedRegion",
    "ProfileObservation",
    "REQUIRED_CHECKS",
    "RESOURCES",
    "Rejection",
    "ValidatedScope",
    "audit_json",
    "choose_partition",
]

RESOURCES: tuple[str, ...] = ("sdnn", "cluster")

# Alphabetical order makes exact-cost ties independent of any speed assumption.
_RESOURCE_ORDER: dict[str, int] = {"cluster": 0, "sdnn": 1}

REQUIRED_CHECKS: tuple[str, ...] = (
    "compile_ok",
    "ordinary_ok",
    "repeat_ok",
    "special_ok",
    "immutable_ok",
)

DEPLOYMENT_ADVISORY: str = ("Scheduling metadata only. Before deployment, the full partition plan "
                            "requires end-to-end numerical validation and performance validation on the "
                            "target backend/context. No runtime autotune or capture occurred, and this "
                            "result is not a production certification.")


@dataclass(frozen=True)
class ProfileObservation:
    """One measured contiguous interval for one resource.

    ``start`` and ``end`` are positional indices into the planner's ordered
    ``node_ids`` sequence.  ``median_us`` must cover this whole interval,
    including output materialization.  ``verified_resources`` is the parent's
    immutable evidence tuple for every generated kernel in the interval; it
    must be nonempty and contain only ``resource``.  External calls and native
    mm nodes without verified evidence must not be represented as SDNN.
    """

    resource: str
    context_key: str
    start: int
    end: int
    median_us: float
    evidence_id: str
    verified_resources: tuple[str, ...]
    compile_ok: bool = False
    ordinary_ok: bool = False
    repeat_ok: bool = False
    special_ok: bool = False
    immutable_ok: bool = False
    external_calls: int = 0
    invalid_reason: str | None = None


@dataclass(frozen=True)
class Rejection:
    """A retained reason for excluding an observation or declining a plan."""

    reason: str
    observation_index: int | None = None
    evidence_id: str | None = None

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "observation_index": self.observation_index,
            "evidence_id": self.evidence_id,
        }


@dataclass(frozen=True)
class PlannedRegion:
    """One selected contiguous interval and its source measurement."""

    start: int
    end: int
    node_ids: tuple[str, ...]
    resource: str
    median_us: float
    evidence_id: str
    context_key: str
    verified_resources: tuple[str, ...]


@dataclass(frozen=True)
class ValidatedScope:
    """Scope over which a plan's observations were validated."""

    context_key: str
    start: int
    end: int
    node_ids: tuple[str, ...]
    checks_passed: tuple[str, ...]


@dataclass(frozen=True)
class PartitionPlan:
    """A complete, contiguous scheduling-only partition recommendation."""

    context_key: str
    regions: tuple[PlannedRegion, ...]
    total_estimated_us: float
    cut_cost_us: float
    validated_scope: ValidatedScope
    rejections: tuple[Rejection, ...]
    advisory: str = DEPLOYMENT_ADVISORY

    @property
    def status(self) -> str:
        return "plan"

    @property
    def total_regions(self) -> int:
        return len(self.regions)

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "status":
            self.status,
            "context_key":
            self.context_key,
            "total_estimated_us":
            self.total_estimated_us,
            "cut_cost_us":
            self.cut_cost_us,
            "validated_scope": {
                "context_key": self.validated_scope.context_key,
                "start": self.validated_scope.start,
                "end": self.validated_scope.end,
                "node_ids": list(self.validated_scope.node_ids),
                "checks_passed": list(self.validated_scope.checks_passed),
            },
            "regions": [{
                "start": region.start,
                "end": region.end,
                "node_ids": list(region.node_ids),
                "resource": region.resource,
                "median_us": region.median_us,
                "evidence_id": region.evidence_id,
                "context_key": region.context_key,
                "verified_resources": list(region.verified_resources),
            } for region in self.regions],
            "rejections": [rejection.to_audit_dict() for rejection in self.rejections],
            "advisory":
            self.advisory,
        }


@dataclass(frozen=True)
class NoPlan:
    """Explicit no-plan result; the parent may keep its native Inductor path."""

    context_key: str
    node_count: int
    rejections: tuple[Rejection, ...]
    advisory: str = DEPLOYMENT_ADVISORY

    @property
    def status(self) -> str:
        return "no_plan"

    def to_audit_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "context_key": self.context_key,
            "node_count": self.node_count,
            "rejections": [rejection.to_audit_dict() for rejection in self.rejections],
            "advisory": self.advisory,
        }


PartitionDecision: TypeAlias = PartitionPlan | NoPlan


@dataclass(frozen=True)
class _Candidate:
    start: int
    end: int
    resource: str
    median_us: float
    evidence_id: str
    context_key: str
    verified_resources: tuple[str, ...]


@dataclass(frozen=True)
class _Partial:
    regions: tuple[_Candidate, ...]
    total_us: float
    cut_us: float


def _validate_node_ids(node_ids: Sequence[str]) -> tuple[str, ...]:
    if isinstance(node_ids, (str, bytes)) or not isinstance(node_ids, Sequence):
        raise TypeError("node_ids must be an ordered sequence of unique node ID strings")
    ordered = tuple(node_ids)
    if not ordered:
        raise ValueError("node_ids must not be empty")
    seen: set[str] = set()
    for index, node_id in enumerate(ordered):
        if not isinstance(node_id, str):
            raise TypeError(f"node_ids[{index}] must be a string")
        if not node_id.strip():
            raise ValueError(f"node_ids[{index}] must be nonempty")
        if node_id in seen:
            raise ValueError(f"duplicate node ID at index {index}: {node_id!r}")
        seen.add(node_id)
    return ordered


def _validate_context_key(context_key: str) -> str:
    if not isinstance(context_key, str):
        raise TypeError("context_key must be a string")
    if not context_key.strip():
        raise ValueError("context_key must be nonempty")
    return context_key


def _validate_max_regions(max_regions: int | None) -> None:
    if max_regions is None:
        return
    if isinstance(max_regions, bool) or not isinstance(max_regions, int):
        raise TypeError("max_regions must be a positive integer when provided")
    if max_regions <= 0:
        raise ValueError("max_regions must be positive")


def _coerce_boundary_cost(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{label} must be a finite non-negative number")
    try:
        cost = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite non-negative number") from exc
    if not math.isfinite(cost) or cost < 0.0:
        raise ValueError(f"{label} must be a finite non-negative number")
    return cost


def _normalize_boundary_costs(
    node_count: int,
    boundary_costs: Sequence[float] | Mapping[int, float] | None,
) -> tuple[float, ...]:
    """Return cuts indexed by the node index immediately before the cut."""
    cut_count = max(node_count - 1, 0)
    costs = [0.0] * cut_count
    if boundary_costs is None:
        return tuple(costs)

    if isinstance(boundary_costs, Mapping):
        for cut_index, value in boundary_costs.items():
            if isinstance(cut_index, bool) or not isinstance(cut_index, int):
                raise TypeError("boundary cost mapping keys must be integer cut indices")
            if cut_index < 0 or cut_index >= cut_count:
                raise ValueError(f"boundary cut index {cut_index} is outside [0, {cut_count - 1}]")
            costs[cut_index] = _coerce_boundary_cost(value, f"boundary_costs[{cut_index}]")
        return tuple(costs)

    if isinstance(boundary_costs, (str, bytes)) or not isinstance(boundary_costs, Sequence):
        raise TypeError("boundary_costs must be a sequence of length node_count - 1, "
                        "a cut-index mapping, or None")
    if len(boundary_costs) != cut_count:
        raise ValueError(f"boundary_costs must have exactly {cut_count} entries, "
                         f"got {len(boundary_costs)}")
    for cut_index, value in enumerate(boundary_costs):
        costs[cut_index] = _coerce_boundary_cost(value, f"boundary_costs[{cut_index}]")
    return tuple(costs)


def _safe_evidence_id(observation: ProfileObservation) -> str | None:
    evidence_id = observation.evidence_id
    if isinstance(evidence_id, str) and evidence_id.strip():
        return evidence_id
    return None


def _observation_rejection_reasons(
    observation: ProfileObservation,
    node_count: int,
    context_key: str,
) -> tuple[str, ...]:
    reasons: list[str] = []

    if observation.invalid_reason is not None:
        if isinstance(observation.invalid_reason, str):
            detail = observation.invalid_reason.strip()
            reasons.append(f"invalid observation: {detail or 'invalid_reason was provided'}")
        else:
            reasons.append("invalid observation: invalid_reason is not a string")

    if not isinstance(observation.context_key, str) or not observation.context_key.strip():
        reasons.append("context_key must be a nonempty string")
    elif observation.context_key != context_key:
        reasons.append("context mismatch: observations are exact and are not extrapolated")

    if not isinstance(observation.resource, str) or observation.resource not in RESOURCES:
        reasons.append(f"resource must be one of {RESOURCES!r}")

    range_is_well_typed = (isinstance(observation.start, int) and not isinstance(observation.start, bool)
                           and isinstance(observation.end, int) and not isinstance(observation.end, bool))
    if not range_is_well_typed:
        reasons.append("range start/end must be integer positional indices")
    elif not (0 <= observation.start < observation.end <= node_count):
        reasons.append(f"range [{observation.start}, {observation.end}) is not a nonempty "
                       f"subrange of [0, {node_count})")

    cost_is_valid = False
    if isinstance(observation.median_us, Real) and not isinstance(observation.median_us, bool):
        try:
            cost = float(observation.median_us)
        except (OverflowError, ValueError):
            cost = math.nan
        cost_is_valid = math.isfinite(cost) and cost > 0.0
    if not cost_is_valid:
        reasons.append("median_us must be a finite positive number")

    if (not isinstance(observation.evidence_id, str) or not observation.evidence_id.strip()):
        reasons.append("evidence_id must be a nonempty string")

    evidence = observation.verified_resources
    if not isinstance(evidence, tuple):
        reasons.append("resource evidence must be a nonempty tuple")
    elif not evidence:
        reasons.append("resource evidence is missing; no verified kernels")
    elif any(not isinstance(item, str) or item != observation.resource for item in evidence):
        reasons.append("resource evidence must contain only the requested resource; "
                       "mixed or unverified evidence is not usable")

    for check_name in REQUIRED_CHECKS:
        if getattr(observation, check_name, None) is not True:
            reasons.append(f"required check {check_name} did not pass")

    if (isinstance(observation.external_calls, bool) or not isinstance(observation.external_calls, int)
            or observation.external_calls != 0):
        reasons.append("external_calls must be the integer 0; external kernels are unverified")

    return tuple(reasons)


def _candidate_sort_key(candidate: _Candidate) -> tuple[Any, ...]:
    return (
        _RESOURCE_ORDER[candidate.resource],
        candidate.end,
        candidate.evidence_id,
        candidate.median_us,
        candidate.context_key,
        candidate.verified_resources,
    )


def _path_sort_key(partial: _Partial) -> tuple[Any, ...]:
    return (
        partial.total_us,
        len(partial.regions),
        tuple(_RESOURCE_ORDER[region.resource] for region in partial.regions),
        tuple((region.start, region.end) for region in partial.regions),
        tuple(region.evidence_id for region in partial.regions),
    )


def _assert_full_coverage(regions: tuple[_Candidate, ...], node_count: int) -> None:
    if not regions or regions[0].start != 0 or regions[-1].end != node_count:
        raise RuntimeError("internal error: plan does not cover the full node range")
    for previous, current in zip(regions, regions[1:]):
        if previous.end != current.start:
            raise RuntimeError("internal error: plan has a gap or overlap")


def choose_partition(
    node_ids: Sequence[str],
    context_key: str,
    observations: Iterable[ProfileObservation],
    boundary_costs: Sequence[float] | Mapping[int, float] | None = None,
    *,
    is_baidu_backend: bool,
    max_regions: int | None = None,
) -> PartitionDecision:
    """Choose an exact minimum-cost contiguous partition, or explicit no-plan.

    ``boundary_costs[i]`` (sequence position or mapping key ``i``) is the
    caller-supplied cost of cutting after node index ``i``.  A missing mapping
    entry defaults to zero and ``None`` means all cuts are zero.  Supplied costs
    already include region materialization; this planner adds no other overhead.

    Validation failures for malformed planning arguments raise ``TypeError`` or
    ``ValueError``.  Bad observations are excluded and their reasons are kept
    in the result's ``rejections`` tuple.  A false Baidu flag yields ``NoPlan``;
    a non-boolean flag raises ``TypeError``.
    """
    ordered_nodes = _validate_node_ids(node_ids)
    context_key = _validate_context_key(context_key)
    if not isinstance(is_baidu_backend, bool):
        raise TypeError("is_baidu_backend must be a bool")
    _validate_max_regions(max_regions)
    cuts = _normalize_boundary_costs(len(ordered_nodes), boundary_costs)

    if isinstance(observations, (str, bytes, ProfileObservation)) or not isinstance(observations, Iterable):
        raise TypeError("observations must be an iterable of ProfileObservation")
    observations_tuple = tuple(observations)

    node_count = len(ordered_nodes)
    rejections: list[Rejection] = []
    candidates: list[_Candidate] = []

    for index, observation in enumerate(observations_tuple):
        if not isinstance(observation, ProfileObservation):
            rejections.append(
                Rejection(
                    reason="observation must be a ProfileObservation instance",
                    observation_index=index,
                ))
            continue
        reasons = _observation_rejection_reasons(observation, node_count, context_key)
        safe_evidence_id = _safe_evidence_id(observation)
        for reason in reasons:
            rejections.append(Rejection(
                reason=reason,
                observation_index=index,
                evidence_id=safe_evidence_id,
            ))
        if reasons:
            continue
        candidates.append(
            _Candidate(
                start=observation.start,
                end=observation.end,
                resource=observation.resource,
                median_us=float(observation.median_us),
                evidence_id=observation.evidence_id,
                context_key=observation.context_key,
                verified_resources=observation.verified_resources,
            ))

    if not is_baidu_backend:
        rejections.append(
            Rejection(reason=("backend: planner is enabled only for a positively identified "
                              "Baidu backend")))
        return NoPlan(
            context_key=context_key,
            node_count=node_count,
            rejections=tuple(rejections),
        )

    if not candidates:
        rejections.append(Rejection(reason="plan: no eligible profile observations after validation"))
        return NoPlan(
            context_key=context_key,
            node_count=node_count,
            rejections=tuple(rejections),
        )

    by_start: list[list[_Candidate]] = [[] for _ in range(node_count)]
    for candidate in candidates:
        by_start[candidate.start].append(candidate)
    for start in range(node_count):
        by_start[start].sort(key=_candidate_sort_key)

    region_limit = node_count if max_regions is None else min(max_regions, node_count)
    # states[end][region_count] -> best partial plan for [0, end).
    states: list[dict[int, _Partial]] = [dict() for _ in range(node_count + 1)]
    states[0][0] = _Partial(regions=(), total_us=0.0, cut_us=0.0)

    for end in range(node_count):
        if not states[end]:
            continue
        starting_here = by_start[end]
        if not starting_here:
            continue
        for region_count, partial in tuple(states[end].items()):
            next_count = region_count + 1
            if next_count > region_limit:
                continue
            cut_cost = 0.0 if end == 0 else cuts[end - 1]
            for candidate in starting_here:
                new_total = partial.total_us + candidate.median_us + cut_cost
                if not math.isfinite(new_total):
                    continue
                new_partial = _Partial(
                    regions=partial.regions + (candidate, ),
                    total_us=new_total,
                    cut_us=partial.cut_us + cut_cost,
                )
                previous = states[candidate.end].get(next_count)
                if previous is None or _path_sort_key(new_partial) < _path_sort_key(previous):
                    states[candidate.end][next_count] = new_partial

    if not states[node_count]:
        limit_text = (f" with max_regions={max_regions}" if max_regions is not None else "")
        rejections.append(
            Rejection(reason=(f"plan: no full contiguous partition of positions "
                              f"[0, {node_count}) was found among eligible observations"
                              f"{limit_text}")))
        return NoPlan(
            context_key=context_key,
            node_count=node_count,
            rejections=tuple(rejections),
        )

    best = min(states[node_count].values(), key=_path_sort_key)
    _assert_full_coverage(best.regions, node_count)

    planned_regions = tuple(
        PlannedRegion(
            start=region.start,
            end=region.end,
            node_ids=ordered_nodes[region.start:region.end],
            resource=region.resource,
            median_us=region.median_us,
            evidence_id=region.evidence_id,
            context_key=region.context_key,
            verified_resources=region.verified_resources,
        ) for region in best.regions)
    scope = ValidatedScope(
        context_key=context_key,
        start=0,
        end=node_count,
        node_ids=ordered_nodes,
        checks_passed=REQUIRED_CHECKS,
    )
    return PartitionPlan(
        context_key=context_key,
        regions=planned_regions,
        total_estimated_us=best.total_us,
        cut_cost_us=best.cut_us,
        validated_scope=scope,
        rejections=tuple(rejections),
    )


def audit_json(decision: PartitionDecision) -> str:
    """Serialize a decision's audit data as strict, JSON-safe text."""
    if not isinstance(decision, (PartitionPlan, NoPlan)):
        raise TypeError("decision must be a PartitionPlan or NoPlan")
    return json.dumps(
        decision.to_audit_dict(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
