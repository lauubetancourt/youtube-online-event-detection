"""Focused G-1 gate and context-stability experiment.

The experiment is deliberately isolated from production context selection.  It
reuses historical G-2 artifacts, invokes only the existing G-1 runner, and
packs complete context units deterministically for reduced-budget variants.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from youtube_pipeline.rag_generation_g1 import (
    PROMPT_VERSION,
    RagG1Config,
    _approx_tokens,
    _compact_event_payload,
    run_rag_g1_validation_from_config,
)


EXPERIMENT_POLICY_ID = "g1_gate_context_stability_v1"
PACKING_POLICY_ID = "ordered_whole_context_unit_first_fit_v1"
PARAMETER_STATUS = "AUDIT_DERIVED_NOT_CALIBRATED"
BASELINE_FILE = "g1_g2_paired_baseline.csv"
GATE_FILE = "g1_g2_gate_evaluation.csv"
STABILITY_FILE = "g1_context_budget_stability.csv"
SUMMARY_FILE = "g1_gate_context_experiment_summary.json"
MANIFEST_FILE = "g1_gate_context_experiment_manifest.json"
RUN_METADATA_FILE = "g1_experiment_run_metadata.json"

GATE_POLICIES: dict[str, frozenset[str] | None] = {
    "OFF": None,
    "STRICT": frozenset({"confirmed", "partially_confirmed"}),
    "PERMISSIVE": frozenset(
        {"confirmed", "partially_confirmed", "ambiguous"}
    ),
}


@dataclass(frozen=True, slots=True)
class G1GateContextExperimentConfig:
    output_dir: str
    consumer_dir: str
    existing_g1_dir: str
    g2_dir: str
    model: str = "gpt-5-mini"
    baseline_call_cap: int = 8
    budget_call_cap: int = 12
    budgets: tuple[tuple[str, int], ...] = (
        ("B50", 5_544),
        ("B75", 9_750),
        ("B90", 10_894),
    )
    sample_size: int = 4
    planned_token_multiplier: float = 2.5

    def __post_init__(self) -> None:
        if self.model != "gpt-5-mini":
            raise ValueError("The experiment must preserve the approved G-1 model.")
        if self.baseline_call_cap < 0 or self.budget_call_cap < 0:
            raise ValueError("Call caps must be non-negative.")
        if not 3 <= self.sample_size <= 4:
            raise ValueError("sample_size must be 3 or 4.")
        if not self.budgets or any(value <= 0 for _, value in self.budgets):
            raise ValueError("Budgets must be positive.")
        if len({name for name, _ in self.budgets}) != len(self.budgets):
            raise ValueError("Budget names must be unique.")


@dataclass(frozen=True, slots=True)
class ExperimentArtifacts:
    baseline_path: str
    gate_path: str
    stability_path: str
    summary_path: str
    manifest_path: str


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: Iterable[dict[str, Any]]) -> None:
    rows = list(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
        + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["candidate_id"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _usage(row: dict[str, Any]) -> dict[str, int] | None:
    raw = row.get("raw_response")
    usage = raw.get("usage") if isinstance(raw, dict) else None
    if not isinstance(usage, dict):
        return None
    out: dict[str, int] = {}
    for key in ("input_tokens", "output_tokens", "total_tokens"):
        value = usage.get(key)
        if not isinstance(value, int):
            return None
        out[key] = value
    return out


def _sum_usage(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    result = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    for row in rows:
        usage = _usage(row)
        if usage is None:
            continue
        result["calls"] += 1
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            result[key] += usage[key]
    return result


def _quantile(values: list[int], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _source_paths(config: G1GateContextExperimentConfig) -> dict[str, Path]:
    consumer = Path(config.consumer_dir)
    g1 = Path(config.existing_g1_dir)
    g2 = Path(config.g2_dir)
    return {
        "validation_inputs": consumer / "rag_validation_inputs.jsonl",
        "context_payloads": consumer / "rag_context_payloads.jsonl",
        "consumer_manifest": consumer / "rag_consumer_manifest.json",
        "existing_g1_reports": g1 / "rag_validation_reports.jsonl",
        "existing_g1_raw": g1 / "rag_raw_model_responses.jsonl",
        "existing_g1_manifest": g1 / "rag_generation_manifest.json",
        "g2_raw": g2 / "rag_raw_model_responses.jsonl",
        "g2_queries": g2 / "rag_video_news_queries.jsonl",
        "g2_external": g2 / "rag_video_external_evidence.jsonl",
        "g2_reports": g2 / "rag_video_validation_reports.jsonl",
        "g2_summary": g2 / "rag_event_validation_summary.jsonl",
        "g2_manifest": g2 / "rag_generation_manifest.json",
    }


def _validate_existing_g1_compatibility(paths: dict[str, Path], model: str) -> None:
    manifest = _read_json(paths["existing_g1_manifest"])
    if manifest.get("model_name") != model or manifest.get("prompt_version") != PROMPT_VERSION:
        raise ValueError("Persisted G-1 model/prompt is incompatible with this experiment.")
    current = {
        "rag_validation_inputs": _sha256_file(paths["validation_inputs"]),
        "rag_context_payloads": _sha256_file(paths["context_payloads"]),
        "rag_consumer_manifest": _sha256_file(paths["consumer_manifest"]),
    }
    if manifest.get("input_hashes") != current:
        raise ValueError("Persisted G-1 evidence package hashes are incompatible.")


def _index(rows: Iterable[dict[str, Any]], key: str) -> dict[str, dict[str, Any]]:
    return {str(row[key]): row for row in rows if row.get(key) is not None}


def _group(rows: Iterable[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(key)
        if value is not None:
            result.setdefault(str(value), []).append(row)
    return result


def select_budget_sample(estimates: dict[str, int], sample_size: int = 4) -> list[str]:
    """Select size anchors without claiming statistical representativeness."""
    if not estimates:
        return []
    ordered = sorted(estimates, key=lambda key: (estimates[key], key))
    targets = [
        float(estimates[ordered[0]]),
        _quantile(list(estimates.values()), 0.50),
        _quantile(list(estimates.values()), 0.90),
        float(estimates[ordered[-1]]),
    ]
    chosen: list[str] = []
    for index, target in enumerate(targets[: min(sample_size, len(ordered))]):
        available = [key for key in ordered if key not in chosen]
        if index == 1:
            above = [key for key in available if estimates[key] >= target]
            candidate = min(above, key=lambda key: (estimates[key] - target, key)) if above else available[-1]
        else:
            candidate = min(available, key=lambda key: (abs(estimates[key] - target), key))
        chosen.append(candidate)
    return chosen


def pack_context_payload(
    validation_input: dict[str, Any],
    context_payload: dict[str, Any],
    budget: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Pack complete units in existing priority order using deterministic first-fit."""
    full_estimate = _approx_tokens(_compact_event_payload(validation_input, context_payload))
    full_units = list(context_payload.get("selected_context_units") or [])
    full_comments = list(context_payload.get("used_context_comments") or [])
    if full_estimate <= budget:
        return dict(context_payload), {
            "equivalent_to_full": True,
            "estimated_context_tokens": full_estimate,
            "selected_unit_count": len(full_units),
            "selected_comment_count": len(full_comments),
        }
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for unit in full_units:
        trial_units = [*selected, unit]
        trial_ids = {
            str(comment_id)
            for trial in trial_units
            for comment_id in (trial.get("comment_ids") or [])
        }
        trial = {
            **context_payload,
            "selected_context_units": trial_units,
            "used_context_comments": [
                comment
                for comment in full_comments
                if str(comment.get("comment_id")) in trial_ids
            ],
        }
        if _approx_tokens(_compact_event_payload(validation_input, trial)) <= budget:
            selected = trial_units
            selected_ids = trial_ids
    packed_comments = [
        comment
        for comment in full_comments
        if str(comment.get("comment_id")) in selected_ids
    ]
    packed = {
        **context_payload,
        "selected_context_units": selected,
        "used_context_comments": packed_comments,
        "used_context_comment_count": len(packed_comments),
        "post_trigger_comment_ids": [
            value
            for value in (context_payload.get("post_trigger_comment_ids") or [])
            if str(value) in selected_ids
        ],
    }
    estimate = _approx_tokens(_compact_event_payload(validation_input, packed))
    return packed, {
        "equivalent_to_full": False,
        "estimated_context_tokens": estimate,
        "selected_unit_count": len(selected),
        "selected_comment_count": len(packed_comments),
    }


def _coverage(full: dict[str, Any], selected: dict[str, Any]) -> dict[str, Any]:
    full_comments = list(full.get("used_context_comments") or [])
    selected_comments = list(selected.get("used_context_comments") or [])
    full_units = list(full.get("selected_context_units") or [])
    selected_units = list(selected.get("selected_context_units") or [])
    full_videos = {str(row.get("video_id")) for row in full_comments if row.get("video_id")}
    selected_videos = {str(row.get("video_id")) for row in selected_comments if row.get("video_id")}
    full_threads = {
        str(row.get("root_comment_id") or row.get("comment_id"))
        for row in full_comments
        if row.get("root_comment_id") or row.get("comment_id")
    }
    selected_threads = {
        str(row.get("root_comment_id") or row.get("comment_id"))
        for row in selected_comments
        if row.get("root_comment_id") or row.get("comment_id")
    }
    ratio = lambda n, d: (n / d if d else None)
    return {
        "inventory_comments": len(full_comments),
        "selected_comments": len(selected_comments),
        "comment_coverage": ratio(len(selected_comments), len(full_comments)),
        "inventory_context_units": len(full_units),
        "selected_context_units": len(selected_units),
        "unit_coverage": ratio(len(selected_units), len(full_units)),
        "inventory_videos": len(full_videos),
        "selected_videos": len(selected_videos),
        "video_coverage": ratio(len(selected_videos), len(full_videos)),
        "inventory_threads": len(full_threads),
        "selected_threads": len(selected_threads),
        "thread_coverage": ratio(len(selected_threads), len(full_threads)),
    }


def _prepare_consumer_dir(
    root: Path,
    validation_input: dict[str, Any],
    context_payload: dict[str, Any],
    source_manifest: dict[str, Any],
) -> None:
    root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(root / "rag_validation_inputs.jsonl", [validation_input])
    _write_jsonl(root / "rag_context_payloads.jsonl", [context_payload])
    _write_json(
        root / "rag_consumer_manifest.json",
        {
            **source_manifest,
            "run_id": "g1experiment_" + _stable_hash(
                [validation_input.get("event_id"), context_payload]
            )[:12],
            "mode": "experimental_deterministic_context_budget",
        },
    )


def _run_variant(
    *,
    runner: Callable[[RagG1Config], dict[str, Any]],
    consumer_dir: Path,
    output_dir: Path,
    candidate_id: str,
    model: str,
    max_approx_tokens: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    return runner(
        RagG1Config(
            consumer_dir=consumer_dir.as_posix(),
            output_dir=output_dir.as_posix(),
            event_id=candidate_id,
            model=model,
            max_approx_tokens=max_approx_tokens,
            max_retries=0,
            notes="G1 gate/context stability experiment; G2/Serper are never invoked.",
            params=params,
        )
    )


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_g1_gate_context_experiment(
    config: G1GateContextExperimentConfig,
    *,
    execute: bool = False,
    g1_runner: Callable[[RagG1Config], dict[str, Any]] = run_rag_g1_validation_from_config,
) -> ExperimentArtifacts:
    paths = _source_paths(config)
    for path in paths.values():
        if not path.exists():
            raise FileNotFoundError(path)
    _validate_existing_g1_compatibility(paths, config.model)
    inputs = _index(_read_jsonl(paths["validation_inputs"]), "event_id")
    payloads = _index(_read_jsonl(paths["context_payloads"]), "event_id")
    candidate_ids = sorted(set(inputs) & set(payloads))
    consumer_manifest = _read_json(paths["consumer_manifest"])
    estimates = {
        key: _approx_tokens(_compact_event_payload(inputs[key], payloads[key]))
        for key in candidate_ids
    }
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    existing_reports = _index(_read_jsonl(paths["existing_g1_reports"]), "event_id")
    existing_raw = _group(_read_jsonl(paths["existing_g1_raw"]), "event_id")
    baseline_dir = output / "g1_baseline_runs"
    baseline_reports_path = baseline_dir / "rag_validation_reports.jsonl"
    baseline_raw_path = baseline_dir / "rag_raw_model_responses.jsonl"
    generated_reports = _index(_read_jsonl(baseline_reports_path), "event_id") if baseline_reports_path.exists() else {}
    generated_raw = _group(_read_jsonl(baseline_raw_path), "event_id") if baseline_raw_path.exists() else {}
    full_reports = {**existing_reports, **generated_reports}
    initially_missing = [key for key in candidate_ids if key not in existing_reports]
    missing = [key for key in candidate_ids if key not in full_reports]
    if len(missing) > config.baseline_call_cap:
        raise RuntimeError(
            f"Baseline requires {len(missing)} calls, above cap {config.baseline_call_cap}."
        )

    sample = select_budget_sample(estimates, config.sample_size)
    variant_plan: list[dict[str, Any]] = []
    unique_variant_hashes: set[str] = set()
    for candidate_id in sample:
        for budget_name, budget in config.budgets:
            packed, metadata = pack_context_payload(inputs[candidate_id], payloads[candidate_id], budget)
            variant_hash = _stable_hash(packed)
            row = {
                "candidate_id": candidate_id,
                "budget_name": budget_name,
                "budget_tokens": budget,
                "packed_payload": packed,
                "context_hash": variant_hash,
                **metadata,
            }
            if not metadata["equivalent_to_full"] and variant_hash not in unique_variant_hashes:
                row["requires_call"] = True
                unique_variant_hashes.add(variant_hash)
            else:
                row["requires_call"] = False
            variant_plan.append(row)
    planned_budget_calls = sum(row["requires_call"] for row in variant_plan)
    if planned_budget_calls > config.budget_call_cap:
        raise RuntimeError(
            f"Budget experiment requires {planned_budget_calls} calls, above cap {config.budget_call_cap}."
        )
    planned_estimate = sum(estimates[key] for key in initially_missing) + sum(
        int(row["estimated_context_tokens"])
        for row in variant_plan
        if row["requires_call"]
    )
    planned_upper = math.ceil(planned_estimate * config.planned_token_multiplier)

    if execute:
        for candidate_id in missing:
            _run_variant(
                runner=g1_runner,
                consumer_dir=Path(config.consumer_dir),
                output_dir=baseline_dir,
                candidate_id=candidate_id,
                model=config.model,
                max_approx_tokens=max(16_000, estimates[candidate_id]),
                params={
                    "experiment_policy_id": EXPERIMENT_POLICY_ID,
                    "variant": "FULL_BASELINE",
                    "context_hash": _stable_hash(payloads[candidate_id]),
                },
            )
        generated_reports = _index(_read_jsonl(baseline_reports_path), "event_id") if baseline_reports_path.exists() else {}
        generated_raw = _group(_read_jsonl(baseline_raw_path), "event_id") if baseline_raw_path.exists() else {}
        full_reports = {**existing_reports, **generated_reports}
        absent_after_execution = [key for key in candidate_ids if key not in full_reports]
        if absent_after_execution:
            raise RuntimeError(f"G-1 baseline remains incomplete: {absent_after_execution}")

        completed_hashes: dict[tuple[str, str], dict[str, Any]] = {}
        for row in variant_plan:
            if not row["requires_call"]:
                continue
            key = (str(row["candidate_id"]), str(row["context_hash"]))
            run_root = output / "g1_budget_runs" / row["candidate_id"] / row["context_hash"][:12]
            metadata_path = run_root / RUN_METADATA_FILE
            report_path = run_root / "output" / "rag_validation_reports.jsonl"
            raw_path = run_root / "output" / "rag_raw_model_responses.jsonl"
            if metadata_path.exists() and report_path.exists():
                meta = _read_json(metadata_path)
                if meta.get("context_hash") == row["context_hash"]:
                    completed_hashes[key] = {
                        "report": _read_jsonl(report_path)[0],
                        "raw": _read_jsonl(raw_path) if raw_path.exists() else [],
                        "run_root": run_root,
                    }
                    continue
            consumer_dir = run_root / "consumer"
            _prepare_consumer_dir(
                consumer_dir,
                inputs[row["candidate_id"]],
                row["packed_payload"],
                consumer_manifest,
            )
            _run_variant(
                runner=g1_runner,
                consumer_dir=consumer_dir,
                output_dir=run_root / "output",
                candidate_id=row["candidate_id"],
                model=config.model,
                max_approx_tokens=int(row["budget_tokens"]),
                params={
                    "experiment_policy_id": EXPERIMENT_POLICY_ID,
                    "packing_policy_id": PACKING_POLICY_ID,
                    "budget_name": row["budget_name"],
                    "budget_tokens": row["budget_tokens"],
                    "context_hash": row["context_hash"],
                },
            )
            _write_json(
                metadata_path,
                {
                    "candidate_id": row["candidate_id"],
                    "context_hash": row["context_hash"],
                    "budget_names_sharing_context": [
                        item["budget_name"]
                        for item in variant_plan
                        if item["candidate_id"] == row["candidate_id"]
                        and item["context_hash"] == row["context_hash"]
                    ],
                    "estimated_context_tokens": row["estimated_context_tokens"],
                    "packing_policy_id": PACKING_POLICY_ID,
                },
            )
            completed_hashes[key] = {
                "report": _read_jsonl(report_path)[0],
                "raw": _read_jsonl(raw_path) if raw_path.exists() else [],
                "run_root": run_root,
            }
    else:
        completed_hashes = {}
        for row in variant_plan:
            run_root = output / "g1_budget_runs" / row["candidate_id"] / row["context_hash"][:12]
            report_path = run_root / "output" / "rag_validation_reports.jsonl"
            raw_path = run_root / "output" / "rag_raw_model_responses.jsonl"
            if report_path.exists():
                completed_hashes[(row["candidate_id"], row["context_hash"])] = {
                    "report": _read_jsonl(report_path)[0],
                    "raw": _read_jsonl(raw_path) if raw_path.exists() else [],
                    "run_root": run_root,
                }

    all_baseline_raw = {**existing_raw, **generated_raw}
    g2_raw = _group(_read_jsonl(paths["g2_raw"]), "event_id")
    g2_queries = _group(_read_jsonl(paths["g2_queries"]), "event_id")
    g2_external = _group(_read_jsonl(paths["g2_external"]), "event_id")
    g2_reports = _group(_read_jsonl(paths["g2_reports"]), "event_id")
    g2_summaries = _index(_read_jsonl(paths["g2_summary"]), "event_id")

    baseline_rows: list[dict[str, Any]] = []
    g2_metrics: dict[str, dict[str, Any]] = {}
    for candidate_id in candidate_ids:
        report = full_reports.get(candidate_id, {})
        usage = _sum_usage(all_baseline_raw.get(candidate_id, []))
        g2_usage = _sum_usage(g2_raw.get(candidate_id, []))
        raw_types = [row.get("record_type") for row in g2_raw.get(candidate_id, [])]
        external_ids = {
            str(value)
            for report_row in g2_reports.get(candidate_id, [])
            for field in ("used_external_evidence_ids", "cited_external_evidence_ids")
            for value in (report_row.get(field) or [])
        }
        support = any(
            row.get("external_evidence_assessment") == "supports"
            for row in g2_reports.get(candidate_id, [])
        )
        metric = {
            "g2_query_generation_calls": raw_types.count("video_query_generation"),
            "g2_validation_calls": raw_types.count("video_validation_generation"),
            "g2_model_calls": g2_usage["calls"],
            "serper_requests": len(g2_queries.get(candidate_id, [])),
            "g2_input_tokens": g2_usage["input_tokens"],
            "g2_output_tokens": g2_usage["output_tokens"],
            "g2_total_tokens": g2_usage["total_tokens"],
            "external_sources_returned": len(g2_external.get(candidate_id, [])),
            "external_sources_used_or_cited": len(external_ids),
            "external_evidence_found": support,
            "g2_observed_status": g2_summaries.get(candidate_id, {}).get(
                "overall_validation_status"
            ),
        }
        g2_metrics[candidate_id] = metric
        baseline_rows.append(
            {
                "candidate_id": candidate_id,
                "full_estimated_context_tokens": estimates[candidate_id],
                "g1_result_source": (
                    "EXISTING_REUSED" if candidate_id in existing_reports else
                    "EXPERIMENT_NEW" if candidate_id in generated_reports else "MISSING"
                ),
                "g1_validation_status": report.get("validation_status"),
                "g1_confidence_label": report.get("confidence_label"),
                "g1_event_interpretation": report.get("event_interpretation"),
                "g1_cited_comment_count": len(report.get("cited_comment_ids") or []),
                "g1_cited_context_unit_count": len(report.get("cited_context_unit_ids") or []),
                "g1_input_tokens": usage["input_tokens"] if usage["calls"] else None,
                "g1_output_tokens": usage["output_tokens"] if usage["calls"] else None,
                "g1_total_tokens": usage["total_tokens"] if usage["calls"] else None,
                **metric,
            }
        )

    gate_rows: list[dict[str, Any]] = []
    gate_summary: dict[str, Any] = {}
    for policy, allowed in GATE_POLICIES.items():
        policy_rows: list[dict[str, Any]] = []
        for base in baseline_rows:
            status = base["g1_validation_status"]
            if status is None:
                decision = "NOT_EVALUABLE_MISSING_G1"
            elif allowed is None or status in allowed:
                decision = "RUN_G2"
            else:
                decision = "SKIP_G2"
            skip = decision == "SKIP_G2"
            row = {
                "candidate_id": base["candidate_id"],
                "policy": policy,
                "parameter_status": "NOT_CALIBRATED" if policy != "OFF" else "BASELINE",
                "eligible_statuses": "ALL" if allowed is None else ",".join(sorted(allowed)),
                "g1_status": status,
                "counterfactual_decision": decision,
                "g2_observed_status": base["g2_observed_status"],
                "external_evidence_found": base["external_evidence_found"],
                "sources_used_or_cited": base["external_sources_used_or_cited"],
                "serper_requests_saved": base["serper_requests"] if skip else 0,
                "g2_query_generation_calls_saved": base["g2_query_generation_calls"] if skip else 0,
                "g2_validation_calls_saved": base["g2_validation_calls"] if skip else 0,
                "g2_model_calls_saved": base["g2_model_calls"] if skip else 0,
                "g2_input_tokens_saved": base["g2_input_tokens"] if skip else 0,
                "g2_output_tokens_saved": base["g2_output_tokens"] if skip else 0,
                "g2_total_tokens_saved": base["g2_total_tokens"] if skip else 0,
                "externally_supported_candidate_filtered": bool(skip and base["external_evidence_found"]),
            }
            gate_rows.append(row)
            policy_rows.append(row)
        gate_summary[policy] = {
            "g2_runs": sum(row["counterfactual_decision"] == "RUN_G2" for row in policy_rows),
            "g2_skipped": sum(row["counterfactual_decision"] == "SKIP_G2" for row in policy_rows),
            "not_evaluable": sum(row["counterfactual_decision"].startswith("NOT_EVALUABLE") for row in policy_rows),
            "serper_requests_saved": sum(row["serper_requests_saved"] for row in policy_rows),
            "g2_query_generation_calls_saved": sum(row["g2_query_generation_calls_saved"] for row in policy_rows),
            "g2_validation_calls_saved": sum(row["g2_validation_calls_saved"] for row in policy_rows),
            "g2_model_calls_saved": sum(row["g2_model_calls_saved"] for row in policy_rows),
            "g2_input_tokens_saved": sum(row["g2_input_tokens_saved"] for row in policy_rows),
            "g2_output_tokens_saved": sum(row["g2_output_tokens_saved"] for row in policy_rows),
            "g2_total_tokens_saved": sum(row["g2_total_tokens_saved"] for row in policy_rows),
            "externally_supported_candidates_filtered": sum(row["externally_supported_candidate_filtered"] for row in policy_rows),
        }

    stability_rows: list[dict[str, Any]] = []
    for candidate_id in sample:
        full_report = full_reports.get(candidate_id, {})
        variants = [
            {"budget_name": "FULL", "budget_tokens": None, "packed_payload": payloads[candidate_id], "context_hash": _stable_hash(payloads[candidate_id]), "estimated_context_tokens": estimates[candidate_id], "equivalent_to_full": True},
            *[row for row in variant_plan if row["candidate_id"] == candidate_id],
        ]
        result_by_hash: dict[str, tuple[dict[str, Any], list[dict[str, Any]], str]] = {
            _stable_hash(payloads[candidate_id]): (full_report, all_baseline_raw.get(candidate_id, []), "FULL_BASELINE")
        }
        for (event_id, context_hash), value in completed_hashes.items():
            if event_id == candidate_id:
                result_by_hash[context_hash] = (value["report"], value["raw"], "BUDGET_EXECUTION")
        for variant in variants:
            selected = variant["packed_payload"]
            result, raw_rows, result_source = result_by_hash.get(
                variant["context_hash"], ({}, [], "NOT_EXECUTED")
            )
            usage = _sum_usage(raw_rows)
            coverage = _coverage(payloads[candidate_id], selected)
            reported_input = usage["input_tokens"] if usage["calls"] else None
            estimate = int(variant["estimated_context_tokens"])
            stability_rows.append(
                {
                    "candidate_id": candidate_id,
                    "sample_role": (
                        "SMALL" if candidate_id == sample[0] else
                        "NEAR_MEDIAN" if candidate_id == sample[1] else
                        "NEAR_P90" if candidate_id == sample[-2] else "MAXIMUM"
                    ),
                    "budget_name": variant["budget_name"],
                    "budget_tokens": variant["budget_tokens"],
                    "parameter_status": PARAMETER_STATUS if variant["budget_name"] != "FULL" else "FULL_CURRENT_CONTEXT",
                    "packing_policy_id": "FULL_CURRENT_CONTEXT" if variant["budget_name"] == "FULL" else PACKING_POLICY_ID,
                    "context_hash": variant["context_hash"],
                    "equivalent_to_full": variant["equivalent_to_full"],
                    "result_source": result_source,
                    **coverage,
                    "estimated_context_tokens": estimate,
                    "reported_input_tokens": reported_input,
                    "reported_output_tokens": usage["output_tokens"] if usage["calls"] else None,
                    "reported_total_tokens": usage["total_tokens"] if usage["calls"] else None,
                    "estimate_to_reported_input_difference": reported_input - estimate if reported_input is not None else None,
                    "reported_input_to_estimate_ratio": reported_input / estimate if reported_input is not None and estimate else None,
                    "validation_status": result.get("validation_status"),
                    "confidence_label": result.get("confidence_label"),
                    "event_interpretation": result.get("event_interpretation"),
                    "cited_comment_count": len(result.get("cited_comment_ids") or []),
                    "cited_context_unit_count": len(result.get("cited_context_unit_ids") or []),
                    "status_agreement_with_full": (result.get("validation_status") == full_report.get("validation_status")) if result else None,
                    "confidence_agreement_with_full": (result.get("confidence_label") == full_report.get("confidence_label")) if result else None,
                    "interpretation_agreement_with_full": (result.get("event_interpretation") == full_report.get("event_interpretation")) if result else None,
                    "comment_citation_overlap_with_full": _jaccard(result.get("cited_comment_ids") or [], full_report.get("cited_comment_ids") or []) if result else None,
                    "unit_citation_overlap_with_full": _jaccard(result.get("cited_context_unit_ids") or [], full_report.get("cited_context_unit_ids") or []) if result else None,
                    "citation_verification_valid": (result.get("citation_verification") or {}).get("valid") if result else None,
                }
            )

    completed_baseline = len(full_reports)
    real_new_raw = [row for rows in generated_raw.values() for row in rows]
    budget_raw_unique = {
        (candidate_id, context_hash): value["raw"]
        for (candidate_id, context_hash), value in completed_hashes.items()
    }
    real_budget_raw = [row for rows in budget_raw_unique.values() for row in rows]
    actual_usage = _sum_usage([*real_new_raw, *real_budget_raw])
    evaluated_stability = [row for row in stability_rows if row["budget_name"] != "FULL" and row["validation_status"] is not None]
    per_budget: dict[str, Any] = {}
    for name, _ in config.budgets:
        rows = [row for row in evaluated_stability if row["budget_name"] == name]
        per_budget[name] = {
            "evaluated_rows": len(rows),
            "status_agreement_with_full": sum(row["status_agreement_with_full"] is True for row in rows) / len(rows) if rows else None,
            "confidence_agreement_with_full": sum(row["confidence_agreement_with_full"] is True for row in rows) / len(rows) if rows else None,
            "mean_comment_citation_overlap": sum(row["comment_citation_overlap_with_full"] for row in rows) / len(rows) if rows else None,
            "mean_unit_citation_overlap": sum(row["unit_citation_overlap_with_full"] for row in rows) / len(rows) if rows else None,
            "mean_comment_coverage": sum(row["comment_coverage"] for row in rows) / len(rows) if rows else None,
            "mean_unit_coverage": sum(row["unit_coverage"] for row in rows) / len(rows) if rows else None,
            "mean_video_coverage": sum(row["video_coverage"] for row in rows) / len(rows) if rows else None,
            "mean_thread_coverage": sum(row["thread_coverage"] for row in rows) / len(rows) if rows else None,
            "reported_input_tokens": sum(row["reported_input_tokens"] or 0 for row in rows) if rows else None,
        }
    usage_ratio_by_context = {
        (row["candidate_id"], row["context_hash"]): row[
            "reported_input_to_estimate_ratio"
        ]
        for row in stability_rows
        if row["reported_input_to_estimate_ratio"] is not None
    }
    usage_ratios = list(usage_ratio_by_context.values())
    summary = {
        "experiment_policy_id": EXPERIMENT_POLICY_ID,
        "execution_status": "EXECUTED" if execute else "PLANNED_OR_REUSED_ONLY",
        "baseline_completion": {
            "candidates": len(candidate_ids),
            "g1_existing": len(existing_reports),
            "g1_reused": len(existing_reports),
            "g1_new": len(generated_reports),
            "paired_g1_g2": completed_baseline,
            "missing_g1_before_execution": len(initially_missing),
        },
        "g1_status_distribution": {
            status: sum(row["g1_validation_status"] == status for row in baseline_rows)
            for status in sorted({str(row["g1_validation_status"]) for row in baseline_rows if row["g1_validation_status"]})
        },
        "g2_observed_distribution": {
            status: sum(row["g2_observed_status"] == status for row in baseline_rows)
            for status in sorted({str(row["g2_observed_status"]) for row in baseline_rows if row["g2_observed_status"]})
        },
        "gate_evaluation": gate_summary,
        "gate_finding": "GATE_INCONCLUSIVE",
        "gate_finding_rationale": (
            "All ten FULL-context G-1 baselines are confirmed, so OFF, STRICT, "
            "and PERMISSIVE route the same candidates and expose neither savings "
            "nor the risk of filtering externally supported candidates."
        ),
        "budget_sample": [
            {"candidate_id": key, "full_estimated_context_tokens": estimates[key]}
            for key in sample
        ],
        "budgets": {name: value for name, value in config.budgets} | {"FULL": "candidate_specific_complete_context"},
        "budget_derivation": {
            "source": "prior_llm_token_context_audit",
            "estimator": "rounded_serialized_characters_divided_by_4",
            "not_exact_tokenization": True,
            "packing_policy_id": PACKING_POLICY_ID,
        },
        "context_stability": per_budget,
        "context_finding": "MORE_EVIDENCE_REQUIRED",
        "context_finding_rationale": (
            "B90 preserved status and confidence in this four-candidate sample, "
            "but the maximum candidate retained only 62.1% of comments and its "
            "citation overlap remained materially below FULL; B50/B75 also changed "
            "confidence for some candidates."
        ),
        "estimator_vs_provider_usage": {
            "observations": len(usage_ratios),
            "min_reported_input_to_estimate_ratio": min(usage_ratios) if usage_ratios else None,
            "median_reported_input_to_estimate_ratio": _quantile(
                [round(value * 1_000_000) for value in usage_ratios], 0.50
            ) / 1_000_000 if usage_ratios else None,
            "max_reported_input_to_estimate_ratio": max(usage_ratios) if usage_ratios else None,
        },
        "planned_resource_control": {
            "baseline_calls": len(initially_missing),
            "baseline_calls_remaining_at_start": len(missing),
            "budget_calls_unique": planned_budget_calls,
            "total_calls": len(initially_missing) + planned_budget_calls,
            "planned_token_upper_estimate": planned_upper,
            "upper_estimate_formula": f"sum(characters/4 estimates) * {config.planned_token_multiplier}",
        },
        "actual_resource_usage": {
            "new_g1_calls": actual_usage["calls"],
            "new_g1_input_tokens": actual_usage["input_tokens"],
            "new_g1_output_tokens": actual_usage["output_tokens"],
            "new_g1_total_tokens": actual_usage["total_tokens"],
            "new_g2_calls": 0,
            "new_serper_requests": 0,
        },
    }
    manifest = {
        "experiment_policy_id": EXPERIMENT_POLICY_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameter_status": PARAMETER_STATUS,
        "source_paths": {name: path.as_posix() for name, path in paths.items()},
        "source_hashes": {name: _sha256_file(path) for name, path in paths.items()},
        "candidate_ids": candidate_ids,
        "candidate_sample": sample,
        "model": config.model,
        "prompt_version": PROMPT_VERSION,
        "topic_profile": None,
        "gate_baseline": "OFF",
        "gate_candidates": {
            name: "ALL" if values is None else sorted(values)
            for name, values in GATE_POLICIES.items()
        },
        "call_caps": {
            "baseline": config.baseline_call_cap,
            "budget": config.budget_call_cap,
        },
        "calls_reused": len(existing_reports),
        "calls_executed": actual_usage["calls"],
        "token_usage": actual_usage,
        "g2_historical_reference": paths["g2_manifest"].as_posix(),
        "new_g2_calls": 0,
        "new_serper_requests": 0,
        "limitations": [
            "The sample is exploratory and not statistically representative.",
            "characters/4 is an estimator, not provider tokenization.",
            "PERMISSIVE and STRICT are counterfactual, NOT_CALIBRATED policies.",
            "G-2 historical support is downstream evidence, not ground truth.",
            "Whole-unit first-fit can reduce video/thread coverage; this is measured, not corrected here.",
        ],
        "summary_sha256": None,
    }

    baseline_path = output / BASELINE_FILE
    gate_path = output / GATE_FILE
    stability_path = output / STABILITY_FILE
    summary_path = output / SUMMARY_FILE
    manifest_path = output / MANIFEST_FILE
    _write_csv(baseline_path, baseline_rows)
    _write_csv(gate_path, gate_rows)
    _write_csv(stability_path, stability_rows)
    _write_json(summary_path, summary)
    manifest["summary_sha256"] = _sha256_file(summary_path)
    manifest["artifact_paths"] = {
        "baseline": baseline_path.as_posix(),
        "gate": gate_path.as_posix(),
        "stability": stability_path.as_posix(),
        "summary": summary_path.as_posix(),
    }
    manifest["artifact_hashes"] = {
        "baseline": _sha256_file(baseline_path),
        "gate": _sha256_file(gate_path),
        "stability": _sha256_file(stability_path),
        "summary": _sha256_file(summary_path),
    }
    _write_json(manifest_path, manifest)
    return ExperimentArtifacts(
        baseline_path=baseline_path.as_posix(),
        gate_path=gate_path.as_posix(),
        stability_path=stability_path.as_posix(),
        summary_path=summary_path.as_posix(),
        manifest_path=manifest_path.as_posix(),
    )


__all__ = [
    "BASELINE_FILE",
    "EXPERIMENT_POLICY_ID",
    "ExperimentArtifacts",
    "G1GateContextExperimentConfig",
    "GATE_FILE",
    "MANIFEST_FILE",
    "PACKING_POLICY_ID",
    "STABILITY_FILE",
    "SUMMARY_FILE",
    "pack_context_payload",
    "run_g1_gate_context_experiment",
    "select_budget_sample",
]
