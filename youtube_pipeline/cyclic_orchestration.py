from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .cyclic_ingestion import INTERVAL_POLICY


APPROVED_CYCLE_STATES = {
    "pending",
    "ready",
    "completed_dry_run",
    "failed_contract_validation",
    "skipped_no_comments",
}

REQUIRED_CYCLE_FIELDS = {
    "simulation_run_id",
    "cycle_id",
    "cycle_index",
    "cycle_run_at_local",
    "cycle_run_at_utc",
    "collection_window_start_local",
    "collection_window_end_local",
    "collection_window_start_utc",
    "collection_window_end_utc",
    "analysis_window_start_local",
    "analysis_window_end_local",
    "analysis_window_start_utc",
    "analysis_window_end_utc",
    "analysis_window_size_days",
    "data_cutoff_local",
    "data_cutoff_utc",
    "timezone",
    "canonical_timezone",
    "simulation_mode",
    "rag_mode",
    "future_leak_count",
    "new_comment_count",
    "cumulative_comment_count",
    "analysis_comment_count",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSONL artifact not found: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object: {path}")
        records.append(payload)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


@dataclass
class CyclicOrchestratorConfig:
    simulation_dir: str | Path
    run_monitoring: bool = False
    run_detection: bool = False
    run_rag: bool = False
    update_cycle_state: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "CyclicOrchestratorConfig":
        config_payload = payload.get("cyclic_ingestion_orchestrator", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Cyclic orchestrator config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown cyclic orchestrator config fields: {unknown}")
        return cls(**config_payload)

    def simulation_path(self) -> Path:
        return Path(self.simulation_dir)

    def validate_c2_scope(self) -> None:
        forbidden = {
            "run_monitoring": self.run_monitoring,
            "run_detection": self.run_detection,
            "run_rag": self.run_rag,
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "C-2 is a dry-run orchestrator only. These flags must remain false: "
                + ", ".join(enabled)
            )


def _validate_manifest_contract(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("simulation_mode") != "cyclic_ingestion_simulation":
        errors.append("online_simulation_manifest simulation_mode must be cyclic_ingestion_simulation.")
    if manifest.get("interval_policy") != INTERVAL_POLICY:
        errors.append(f"interval_policy must be {INTERVAL_POLICY}.")
    temporal_policy = manifest.get("temporal_policy", {})
    if not isinstance(temporal_policy, dict):
        errors.append("temporal_policy must be an object.")
    else:
        expected_rules = {
            "no_future_leakage_rule": "event_time_utc < data_cutoff_utc",
            "collection_window_rule": (
                "collection_window_start_utc <= event_time_utc < "
                "collection_window_end_utc"
            ),
            "analysis_window_rule": (
                "analysis_window_start_utc <= event_time_utc < analysis_window_end_utc"
            ),
        }
        for key, expected in expected_rules.items():
            if temporal_policy.get(key) != expected:
                errors.append(f"temporal_policy.{key} must be '{expected}'.")
        if temporal_policy.get("filtering_uses_utc") is not True:
            errors.append("temporal_policy.filtering_uses_utc must be true.")
    return errors


def _validate_required_cycle_fields(cycles: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    for index, cycle in enumerate(cycles, start=1):
        missing = sorted(REQUIRED_CYCLE_FIELDS - set(cycle))
        if missing:
            errors.append(f"Cycle record {index} missing required fields: {missing}")
    return errors


def _validate_cycle_ids(cycles: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for cycle in cycles:
        cycle_id = str(cycle.get("cycle_id"))
        if cycle_id in seen:
            duplicates.add(cycle_id)
        seen.add(cycle_id)
    if duplicates:
        return [f"Duplicate cycle_id values found: {sorted(duplicates)}"]
    return []


def _parse_utc_column(values: list[Any], field_name: str) -> pd.Series:
    parsed = pd.to_datetime(pd.Series(values), utc=True, errors="coerce")
    if parsed.isna().any():
        raise ValueError(f"Field {field_name} contains invalid UTC timestamps.")
    return parsed


def _validate_cycle_order(cycles: list[dict[str, Any]]) -> list[str]:
    if not cycles:
        return []
    cycle_indices = [cycle.get("cycle_index") for cycle in cycles]
    try:
        numeric_indices = [int(value) for value in cycle_indices]
    except (TypeError, ValueError):
        return ["cycle_index must be integer-like for every cycle."]
    run_times = _parse_utc_column(
        [cycle.get("cycle_run_at_utc") for cycle in cycles],
        "cycle_run_at_utc",
    )
    source_order = [
        (numeric_indices[i], run_times.iloc[i].isoformat(), str(cycles[i].get("cycle_id")))
        for i in range(len(cycles))
    ]
    sorted_order = sorted(source_order)
    errors: list[str] = []
    if source_order != sorted_order:
        errors.append("cycle_manifest.jsonl is not ordered by cycle_index, cycle_run_at_utc, cycle_id.")
    if len(set(numeric_indices)) != len(numeric_indices):
        errors.append("cycle_index values must be unique.")
    if not run_times.is_monotonic_increasing:
        errors.append("cycle_run_at_utc must be monotonic increasing.")
    return errors


def _validate_cycle_temporal_rules(cycles: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    if not cycles:
        return errors
    start_collection = _parse_utc_column(
        [cycle.get("collection_window_start_utc") for cycle in cycles],
        "collection_window_start_utc",
    )
    end_collection = _parse_utc_column(
        [cycle.get("collection_window_end_utc") for cycle in cycles],
        "collection_window_end_utc",
    )
    start_analysis = _parse_utc_column(
        [cycle.get("analysis_window_start_utc") for cycle in cycles],
        "analysis_window_start_utc",
    )
    end_analysis = _parse_utc_column(
        [cycle.get("analysis_window_end_utc") for cycle in cycles],
        "analysis_window_end_utc",
    )
    data_cutoff = _parse_utc_column(
        [cycle.get("data_cutoff_utc") for cycle in cycles],
        "data_cutoff_utc",
    )
    if (start_collection >= end_collection).any():
        errors.append("Every collection window must satisfy start_utc < end_utc.")
    if (start_analysis >= end_analysis).any():
        errors.append("Every analysis window must satisfy start_utc < end_utc.")
    if (data_cutoff != end_collection).any():
        errors.append("data_cutoff_utc must equal collection_window_end_utc in C-0/C-1.")
    for cycle in cycles:
        try:
            future_leak_count = int(cycle.get("future_leak_count", 0))
        except (TypeError, ValueError):
            errors.append(f"Cycle {cycle.get('cycle_id')} has invalid future_leak_count.")
            continue
        if future_leak_count != 0:
            errors.append(
                f"Cycle {cycle.get('cycle_id')} has future_leak_count={future_leak_count}."
            )
    return errors


def validate_cycle_contracts(
    *,
    manifest: dict[str, Any],
    cycles: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    errors.extend(_validate_manifest_contract(manifest))
    errors.extend(_validate_required_cycle_fields(cycles))
    errors.extend(_validate_cycle_ids(cycles))
    errors.extend(_validate_cycle_order(cycles))
    errors.extend(_validate_cycle_temporal_rules(cycles))
    return errors


def _status_for_cycle(cycle: dict[str, Any]) -> tuple[str, list[str]]:
    try:
        analysis_comment_count = int(cycle.get("analysis_comment_count", 0))
    except (TypeError, ValueError):
        return "failed_contract_validation", ["pending", "failed_contract_validation"]
    if analysis_comment_count <= 0:
        return "skipped_no_comments", ["pending", "skipped_no_comments"]
    return "completed_dry_run", ["pending", "ready", "completed_dry_run"]


def build_orchestration_plan(cycles: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_cycles = sorted(
        cycles,
        key=lambda cycle: (
            int(cycle["cycle_index"]),
            pd.Timestamp(cycle["cycle_run_at_utc"]),
            str(cycle["cycle_id"]),
        ),
    )
    plan: list[dict[str, Any]] = []
    for order_index, cycle in enumerate(ordered_cycles, start=1):
        final_status, transitions = _status_for_cycle(cycle)
        ready = "ready" in transitions
        record = {
            "simulation_run_id": cycle["simulation_run_id"],
            "cycle_id": cycle["cycle_id"],
            "cycle_index": int(cycle["cycle_index"]),
            "execution_order": order_index,
            "cycle_run_at_local": cycle["cycle_run_at_local"],
            "cycle_run_at_utc": cycle["cycle_run_at_utc"],
            "collection_window_start_utc": cycle["collection_window_start_utc"],
            "collection_window_end_utc": cycle["collection_window_end_utc"],
            "analysis_window_start_utc": cycle["analysis_window_start_utc"],
            "analysis_window_end_utc": cycle["analysis_window_end_utc"],
            "data_cutoff_utc": cycle["data_cutoff_utc"],
            "new_comment_count": int(cycle.get("new_comment_count", 0)),
            "cumulative_comment_count": int(cycle.get("cumulative_comment_count", 0)),
            "analysis_comment_count": int(cycle.get("analysis_comment_count", 0)),
            "future_leak_count": int(cycle.get("future_leak_count", 0)),
            "initial_status": "pending",
            "ready_status": "ready" if ready else None,
            "final_status": final_status,
            "state_transitions": transitions,
            "run_monitoring": False,
            "run_detection": False,
            "run_rag": False,
            "dry_run_only": True,
            "skip_reason": "no_comments_in_analysis_window"
            if final_status == "skipped_no_comments"
            else None,
        }
        unknown_states = set(transitions + [final_status]) - APPROVED_CYCLE_STATES
        if unknown_states:
            raise ValueError(f"Orchestration produced unsupported states: {unknown_states}")
        plan.append(record)
    return plan


def _build_orchestration_manifest(
    *,
    simulation_manifest: dict[str, Any],
    plan: list[dict[str, Any]],
    contract_errors: list[str],
) -> dict[str, Any]:
    status_counts = {
        state: sum(1 for row in plan if row["final_status"] == state)
        for state in sorted(APPROVED_CYCLE_STATES)
    }
    ready_count = sum(1 for row in plan if row["ready_status"] == "ready")
    return {
        "simulation_run_id": simulation_manifest.get("simulation_run_id"),
        "simulation_mode": "cyclic_ingestion_simulation",
        "orchestration_stage": "C-2",
        "orchestration_mode": "dry_run",
        "orchestration_status": "completed_dry_run" if not contract_errors else "failed",
        "contract_validation_status": "passed" if not contract_errors else "failed",
        "contract_errors": contract_errors,
        "interval_policy": simulation_manifest.get("interval_policy"),
        "deterministic_order": [
            "cycle_index asc",
            "cycle_run_at_utc asc",
            "cycle_id asc",
        ],
        "execution_guards": {
            "run_monitoring": False,
            "run_detection": False,
            "run_rag": False,
            "llm_calls": 0,
            "serper_calls": 0,
            "embeddings": False,
            "vectorstore": False,
        },
        "cycle_counts": {
            "total": len(plan),
            "ready": ready_count,
            "completed_dry_run": status_counts.get("completed_dry_run", 0),
            "skipped_no_comments": status_counts.get("skipped_no_comments", 0),
            "failed_contract_validation": status_counts.get(
                "failed_contract_validation", 0
            ),
        },
        "status_counts": status_counts,
        "input_artifacts": {
            "online_simulation_manifest": "online_simulation_manifest.json",
            "cycle_manifest": "cycle_manifest.jsonl",
            "cycle_state": "cycle_state.json",
        },
        "output_artifacts": {
            "cycle_orchestration_manifest": "cycle_orchestration_manifest.json",
            "cycle_orchestration_plan": "cycle_orchestration_plan.jsonl",
            "cycle_state": "cycle_state.json",
        },
        "compatibility": {
            "retrospective_replay_status": "untouched",
            "monitoring_status": "not_executed_c2",
            "detection_status": "not_executed_c2",
            "rag_status": "not_executed_c2",
        },
    }


def _update_cycle_state(
    *,
    state_path: Path,
    orchestration_manifest: dict[str, Any],
    plan: list[dict[str, Any]],
) -> dict[str, Any]:
    state = _read_json(state_path)
    existing_history = state.get("orchestration_history", [])
    if not isinstance(existing_history, list):
        existing_history = []
    state["orchestration"] = {
        "stage": "C-2",
        "status": orchestration_manifest["orchestration_status"],
        "manifest_ref": "cycle_orchestration_manifest.json",
        "plan_ref": "cycle_orchestration_plan.jsonl",
        "last_orchestrated_cycle_id": plan[-1]["cycle_id"] if plan else None,
        "pending_cycle_count": 0,
        "ready_cycle_count": orchestration_manifest["cycle_counts"]["ready"],
        "completed_dry_run_cycle_count": orchestration_manifest["cycle_counts"][
            "completed_dry_run"
        ],
        "failed_cycle_count": orchestration_manifest["cycle_counts"][
            "failed_contract_validation"
        ],
        "run_monitoring": False,
        "run_detection": False,
        "run_rag": False,
    }
    existing_history.append(
        {
            "stage": "C-2",
            "status": orchestration_manifest["orchestration_status"],
            "manifest_ref": "cycle_orchestration_manifest.json",
            "plan_ref": "cycle_orchestration_plan.jsonl",
        }
    )
    state["orchestration_history"] = existing_history
    _write_json(state_path, state)
    return state


def run_cyclic_orchestrator_dry_run(config: CyclicOrchestratorConfig) -> dict[str, Any]:
    config.validate_c2_scope()
    simulation_dir = config.simulation_path()
    manifest_path = simulation_dir / "online_simulation_manifest.json"
    cycle_manifest_path = simulation_dir / "cycle_manifest.jsonl"
    state_path = simulation_dir / "cycle_state.json"
    output_manifest_path = simulation_dir / "cycle_orchestration_manifest.json"
    output_plan_path = simulation_dir / "cycle_orchestration_plan.jsonl"

    simulation_manifest = _read_json(manifest_path)
    cycles = _read_jsonl(cycle_manifest_path)
    contract_errors = validate_cycle_contracts(
        manifest=simulation_manifest,
        cycles=cycles,
    )
    if contract_errors:
        failed_manifest = _build_orchestration_manifest(
            simulation_manifest=simulation_manifest,
            plan=[],
            contract_errors=contract_errors,
        )
        _write_json(output_manifest_path, failed_manifest)
        raise ValueError("Cyclic orchestration contract validation failed: " + "; ".join(contract_errors))

    plan = build_orchestration_plan(cycles)
    orchestration_manifest = _build_orchestration_manifest(
        simulation_manifest=simulation_manifest,
        plan=plan,
        contract_errors=[],
    )
    _write_jsonl(output_plan_path, plan)
    _write_json(output_manifest_path, orchestration_manifest)
    if config.update_cycle_state:
        _update_cycle_state(
            state_path=state_path,
            orchestration_manifest=orchestration_manifest,
            plan=plan,
        )

    return {
        "simulation_run_id": orchestration_manifest["simulation_run_id"],
        "simulation_dir": str(simulation_dir),
        "orchestration_status": orchestration_manifest["orchestration_status"],
        "cycles_total": orchestration_manifest["cycle_counts"]["total"],
        "ready_cycle_count": orchestration_manifest["cycle_counts"]["ready"],
        "completed_dry_run_cycle_count": orchestration_manifest["cycle_counts"][
            "completed_dry_run"
        ],
        "skipped_no_comments_cycle_count": orchestration_manifest["cycle_counts"][
            "skipped_no_comments"
        ],
        "failed_contract_validation_cycle_count": orchestration_manifest[
            "cycle_counts"
        ]["failed_contract_validation"],
        "artifacts": {
            "cycle_orchestration_manifest": str(output_manifest_path),
            "cycle_orchestration_plan": str(output_plan_path),
            "cycle_state": str(state_path),
        },
    }


__all__ = [
    "APPROVED_CYCLE_STATES",
    "CyclicOrchestratorConfig",
    "build_orchestration_plan",
    "run_cyclic_orchestrator_dry_run",
    "validate_cycle_contracts",
]
