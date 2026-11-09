from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .cyclic_ingestion import INTERVAL_POLICY


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
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL line {line_number} must contain an object: {path}")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required CSV artifact not found: {path}")
    return pd.read_csv(path)


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _hash_ids(ids: list[str] | set[str]) -> str:
    joined = "\n".join(sorted(str(value) for value in ids))
    return "sha1_" + hashlib.sha1(joined.encode("utf-8")).hexdigest()


def _metadata_by_comment(input_inventory: pd.DataFrame) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in input_inventory.sort_values(["event_time_utc", "comment_id"]).to_dict(
        orient="records"
    ):
        comment_id = str(row["comment_id"])
        metadata.setdefault(comment_id, row)
    return metadata


@dataclass
class CyclicStatefulAdapterConfig:
    simulation_dir: str | Path
    run_monitoring: bool = False
    run_detection: bool = False
    run_rag: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "CyclicStatefulAdapterConfig":
        config_payload = payload.get("cyclic_stateful_adapter", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Cyclic stateful adapter config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown cyclic stateful adapter config fields: {unknown}")
        return cls(**config_payload)

    def simulation_path(self) -> Path:
        return Path(self.simulation_dir)

    def validate_c3_scope(self) -> None:
        forbidden = {
            "run_monitoring": self.run_monitoring,
            "run_detection": self.run_detection,
            "run_rag": self.run_rag,
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                "C-3 prepares stateful inputs only. These flags must remain false: "
                + ", ".join(enabled)
            )


def _validate_input_artifacts(
    *,
    simulation_manifest: dict[str, Any],
    orchestration_manifest: dict[str, Any],
    orchestration_plan: list[dict[str, Any]],
    input_inventory: pd.DataFrame,
    processed_inventory: pd.DataFrame,
) -> list[str]:
    errors: list[str] = []
    if simulation_manifest.get("simulation_mode") != "cyclic_ingestion_simulation":
        errors.append("online_simulation_manifest must use cyclic_ingestion_simulation.")
    if simulation_manifest.get("interval_policy") != INTERVAL_POLICY:
        errors.append(f"interval_policy must be {INTERVAL_POLICY}.")
    if orchestration_manifest.get("orchestration_status") != "completed_dry_run":
        errors.append("cycle_orchestration_manifest must be completed_dry_run.")
    if orchestration_manifest.get("execution_guards", {}).get("run_detection") is not False:
        errors.append("C-2 run_detection guard must be false.")
    if orchestration_manifest.get("execution_guards", {}).get("run_monitoring") is not False:
        errors.append("C-2 run_monitoring guard must be false.")
    if orchestration_manifest.get("execution_guards", {}).get("run_rag") is not False:
        errors.append("C-2 run_rag guard must be false.")
    if not orchestration_plan:
        errors.append("cycle_orchestration_plan.jsonl must contain at least one cycle.")
    required_input = {
        "simulation_run_id",
        "comment_id",
        "video_id",
        "event_time_utc",
        "assigned_cycle_id",
        "first_seen_cycle_id",
        "is_new_in_cycle",
        "is_duplicate",
    }
    missing_input = sorted(required_input - set(input_inventory.columns))
    if missing_input:
        errors.append(f"cycle_input_inventory.csv missing columns: {missing_input}")
    required_processed = {
        "simulation_run_id",
        "cycle_id",
        "cycle_index",
        "comment_id",
        "video_id",
        "event_time_utc",
        "first_seen_cycle_id",
        "analysis_window_start_utc",
        "analysis_window_end_utc",
        "data_cutoff_utc",
    }
    missing_processed = sorted(required_processed - set(processed_inventory.columns))
    if missing_processed:
        errors.append(f"cycle_processed_inventory.csv missing columns: {missing_processed}")
    return errors


def _ordered_ready_cycles(orchestration_plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cycles = sorted(
        orchestration_plan,
        key=lambda row: (
            int(row["cycle_index"]),
            pd.Timestamp(row["cycle_run_at_utc"]),
            str(row["cycle_id"]),
        ),
    )
    for row in cycles:
        if row.get("final_status") not in {"completed_dry_run", "skipped_no_comments"}:
            raise ValueError(
                f"Cycle {row.get('cycle_id')} is not ready for C-3: {row.get('final_status')}"
            )
        if row.get("run_detection") or row.get("run_monitoring") or row.get("run_rag"):
            raise ValueError(f"Cycle {row.get('cycle_id')} has forbidden execution flags.")
    return cycles


def _active_rows_for_cycle(processed_inventory: pd.DataFrame, cycle_id: str) -> pd.DataFrame:
    return processed_inventory.loc[processed_inventory["cycle_id"] == cycle_id].copy()


def _new_rows_for_cycle(input_inventory: pd.DataFrame, cycle_id: str) -> pd.DataFrame:
    return input_inventory.loc[
        (_bool_series(input_inventory["is_new_in_cycle"]))
        & (~_bool_series(input_inventory["is_duplicate"]))
        & (input_inventory["assigned_cycle_id"] == cycle_id)
    ].copy()


def _exited_rows(
    *,
    exited_ids: set[str],
    metadata: dict[str, dict[str, Any]],
    cycle: dict[str, Any],
    simulation_run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comment_id in sorted(exited_ids):
        meta = metadata.get(comment_id, {})
        rows.append(
            {
                "simulation_run_id": simulation_run_id,
                "cycle_id": cycle["cycle_id"],
                "cycle_index": int(cycle["cycle_index"]),
                "comment_id": comment_id,
                "video_id": meta.get("video_id"),
                "author_id": meta.get("author_id"),
                "event_time_utc": meta.get("event_time_utc"),
                "first_seen_cycle_id": meta.get("first_seen_cycle_id"),
                "analysis_window_start_utc": cycle["analysis_window_start_utc"],
                "analysis_window_end_utc": cycle["analysis_window_end_utc"],
                "data_cutoff_utc": cycle["data_cutoff_utc"],
                "window_membership_role": "exited_window",
                "is_new_in_cycle": False,
                "is_active_in_window": False,
                "is_accumulated_by_cycle": True,
                "exited_window": True,
            }
        )
    return rows


def _active_rows(
    *,
    active_df: pd.DataFrame,
    new_ids: set[str],
    cumulative_ids: set[str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in active_df.sort_values(["event_time_utc", "comment_id"]).to_dict(orient="records"):
        comment_id = str(row["comment_id"])
        is_new = comment_id in new_ids
        role = "active_new" if is_new else "active_existing"
        rows.append(
            {
                "simulation_run_id": row.get("simulation_run_id"),
                "cycle_id": row.get("cycle_id"),
                "cycle_index": int(row.get("cycle_index")),
                "comment_id": comment_id,
                "video_id": row.get("video_id"),
                "author_id": row.get("author_id"),
                "event_time_utc": row.get("event_time_utc"),
                "first_seen_cycle_id": row.get("first_seen_cycle_id"),
                "analysis_window_start_utc": row.get("analysis_window_start_utc"),
                "analysis_window_end_utc": row.get("analysis_window_end_utc"),
                "data_cutoff_utc": row.get("data_cutoff_utc"),
                "window_membership_role": role,
                "is_new_in_cycle": is_new,
                "is_active_in_window": True,
                "is_accumulated_by_cycle": comment_id in cumulative_ids,
                "exited_window": False,
            }
        )
    return rows


def _new_outside_window_rows(
    *,
    new_df: pd.DataFrame,
    active_ids: set[str],
    cycle: dict[str, Any],
    simulation_run_id: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing_df = new_df.loc[~new_df["comment_id"].astype(str).isin(active_ids)]
    for row in missing_df.to_dict(orient="records"):
        rows.append(
            {
                "simulation_run_id": simulation_run_id,
                "cycle_id": cycle["cycle_id"],
                "cycle_index": int(cycle["cycle_index"]),
                "comment_id": str(row.get("comment_id")),
                "video_id": row.get("video_id"),
                "author_id": row.get("author_id"),
                "event_time_utc": row.get("event_time_utc"),
                "first_seen_cycle_id": row.get("first_seen_cycle_id"),
                "analysis_window_start_utc": cycle["analysis_window_start_utc"],
                "analysis_window_end_utc": cycle["analysis_window_end_utc"],
                "data_cutoff_utc": cycle["data_cutoff_utc"],
                "window_membership_role": "new_outside_active_window",
                "is_new_in_cycle": True,
                "is_active_in_window": False,
                "is_accumulated_by_cycle": True,
                "exited_window": False,
            }
        )
    return rows


def _validate_cycle_temporal_semantics(active_df: pd.DataFrame, cycle: dict[str, Any]) -> dict[str, Any]:
    if active_df.empty:
        return {
            "future_leak_count": 0,
            "outside_analysis_window_count": 0,
            "temporal_status": "passed",
        }
    event_time = pd.to_datetime(active_df["event_time_utc"], utc=True, errors="coerce")
    start = pd.Timestamp(cycle["analysis_window_start_utc"])
    end = pd.Timestamp(cycle["analysis_window_end_utc"])
    cutoff = pd.Timestamp(cycle["data_cutoff_utc"])
    future_leak_count = int((event_time >= cutoff).sum())
    outside_count = int(((event_time < start) | (event_time >= end)).sum())
    status = "passed" if future_leak_count == 0 and outside_count == 0 else "failed"
    return {
        "future_leak_count": future_leak_count,
        "outside_analysis_window_count": outside_count,
        "temporal_status": status,
    }


def _build_monitoring_input(
    *,
    cycle: dict[str, Any],
    active_ids: set[str],
    new_ids: set[str],
    exited_ids: set[str],
    cumulative_ids: set[str],
    active_video_count: int,
) -> dict[str, Any]:
    return {
        "simulation_run_id": cycle["simulation_run_id"],
        "cycle_id": cycle["cycle_id"],
        "cycle_index": int(cycle["cycle_index"]),
        "cycle_run_at_utc": cycle["cycle_run_at_utc"],
        "analysis_window_start_utc": cycle["analysis_window_start_utc"],
        "analysis_window_end_utc": cycle["analysis_window_end_utc"],
        "data_cutoff_utc": cycle["data_cutoff_utc"],
        "input_kind": "monitoring_prepared_not_executed",
        "active_window_comment_count": len(active_ids),
        "new_comment_count": len(new_ids),
        "exited_window_comment_count": len(exited_ids),
        "cumulative_comment_count": len(cumulative_ids),
        "active_video_count": active_video_count,
        "active_comment_ids_hash": _hash_ids(active_ids),
        "new_comment_ids_hash": _hash_ids(new_ids),
        "inventory_ref": "cycle_window_inventory.csv",
        "run_monitoring": False,
    }


def _build_detection_input(
    *,
    cycle: dict[str, Any],
    active_ids: set[str],
    new_ids: set[str],
    exited_ids: set[str],
    cumulative_ids: set[str],
    overlap_ids: set[str],
) -> dict[str, Any]:
    return {
        "simulation_run_id": cycle["simulation_run_id"],
        "cycle_id": cycle["cycle_id"],
        "cycle_index": int(cycle["cycle_index"]),
        "cycle_run_at_utc": cycle["cycle_run_at_utc"],
        "analysis_window_start_utc": cycle["analysis_window_start_utc"],
        "analysis_window_end_utc": cycle["analysis_window_end_utc"],
        "data_cutoff_utc": cycle["data_cutoff_utc"],
        "input_kind": "detection_prepared_not_executed",
        "active_window_comment_count": len(active_ids),
        "new_comment_count": len(new_ids),
        "exited_window_comment_count": len(exited_ids),
        "cumulative_comment_count": len(cumulative_ids),
        "overlap_with_previous_cycle_count": len(overlap_ids),
        "active_comment_ids_hash": _hash_ids(active_ids),
        "overlap_comment_ids_hash": _hash_ids(overlap_ids),
        "decision_state_stub_ref": "cycle_stateful_context.json#decision_state_stub",
        "cooldown_state_stub_ref": "cycle_stateful_context.json#cooldown_state_stub",
        "emitted_event_registry_stub_ref": (
            "cycle_stateful_context.json#emitted_event_registry_stub"
        ),
        "inventory_ref": "cycle_window_inventory.csv",
        "run_detection": False,
    }


def _build_readiness_row(
    *,
    cycle: dict[str, Any],
    active_ids: set[str],
    new_ids: set[str],
    exited_ids: set[str],
    temporal: dict[str, Any],
    active_count_matches_processed: bool,
    new_ids_unique_so_far: bool,
) -> dict[str, Any]:
    checks = {
        "has_comment_id_traceability": True,
        "active_count_matches_processed_inventory": active_count_matches_processed,
        "new_ids_unique_so_far": new_ids_unique_so_far,
        "future_leak_count": temporal["future_leak_count"],
        "outside_analysis_window_count": temporal["outside_analysis_window_count"],
        "run_monitoring": False,
        "run_detection": False,
        "run_rag": False,
    }
    readiness_status = (
        "ready_for_future_monitoring_detection"
        if (
            active_count_matches_processed
            and new_ids_unique_so_far
            and temporal["temporal_status"] == "passed"
        )
        else "failed_readiness"
    )
    return {
        "simulation_run_id": cycle["simulation_run_id"],
        "cycle_id": cycle["cycle_id"],
        "cycle_index": int(cycle["cycle_index"]),
        "readiness_status": readiness_status,
        "active_window_comment_count": len(active_ids),
        "new_comment_count": len(new_ids),
        "exited_window_comment_count": len(exited_ids),
        "checks": checks,
    }


def run_cyclic_stateful_adapter(config: CyclicStatefulAdapterConfig) -> dict[str, Any]:
    config.validate_c3_scope()
    simulation_dir = config.simulation_path()
    simulation_manifest = _read_json(simulation_dir / "online_simulation_manifest.json")
    orchestration_manifest = _read_json(
        simulation_dir / "cycle_orchestration_manifest.json"
    )
    orchestration_plan = _read_jsonl(simulation_dir / "cycle_orchestration_plan.jsonl")
    cycle_state = _read_json(simulation_dir / "cycle_state.json")
    input_inventory = _read_csv(simulation_dir / "cycle_input_inventory.csv")
    processed_inventory = _read_csv(simulation_dir / "cycle_processed_inventory.csv")

    input_errors = _validate_input_artifacts(
        simulation_manifest=simulation_manifest,
        orchestration_manifest=orchestration_manifest,
        orchestration_plan=orchestration_plan,
        input_inventory=input_inventory,
        processed_inventory=processed_inventory,
    )
    if input_errors:
        raise ValueError("C-3 input validation failed: " + "; ".join(input_errors))

    cycles = _ordered_ready_cycles(orchestration_plan)
    input_inventory = input_inventory.copy()
    processed_inventory = processed_inventory.copy()
    input_inventory["comment_id"] = input_inventory["comment_id"].astype(str)
    processed_inventory["comment_id"] = processed_inventory["comment_id"].astype(str)
    metadata = _metadata_by_comment(input_inventory)

    simulation_run_id = str(simulation_manifest["simulation_run_id"])
    previous_active_ids: set[str] = set()
    seen_comment_ids: set[str] = set()
    new_seen_once: set[str] = set()

    new_by_cycle: dict[str, list[str]] = {}
    active_by_cycle: dict[str, list[str]] = {}
    exited_by_cycle: dict[str, list[str]] = {}
    cumulative_count_by_cycle: dict[str, int] = {}
    completed_cycle_ids: list[str] = []
    window_rows: list[dict[str, Any]] = []
    monitoring_inputs: list[dict[str, Any]] = []
    detection_inputs: list[dict[str, Any]] = []
    readiness_rows: list[dict[str, Any]] = []
    overlap_cycle_count = 0
    exited_total = 0

    for cycle in cycles:
        cycle_id = cycle["cycle_id"]
        active_df = _active_rows_for_cycle(processed_inventory, cycle_id)
        new_df = _new_rows_for_cycle(input_inventory, cycle_id)
        active_ids = set(active_df["comment_id"].astype(str).tolist())
        new_ids = set(new_df["comment_id"].astype(str).tolist())
        duplicate_new_ids = new_seen_once.intersection(new_ids)
        new_seen_once.update(new_ids)
        seen_comment_ids.update(new_ids)
        exited_ids = previous_active_ids - active_ids
        overlap_ids = previous_active_ids.intersection(active_ids)
        if overlap_ids:
            overlap_cycle_count += 1
        exited_total += len(exited_ids)

        temporal = _validate_cycle_temporal_semantics(active_df, cycle)
        active_video_count = int(active_df["video_id"].nunique()) if not active_df.empty else 0
        window_rows.extend(
            _active_rows(
                active_df=active_df,
                new_ids=new_ids,
                cumulative_ids=seen_comment_ids,
            )
        )
        window_rows.extend(
            _new_outside_window_rows(
                new_df=new_df,
                active_ids=active_ids,
                cycle=cycle,
                simulation_run_id=simulation_run_id,
            )
        )
        window_rows.extend(
            _exited_rows(
                exited_ids=exited_ids,
                metadata=metadata,
                cycle=cycle,
                simulation_run_id=simulation_run_id,
            )
        )

        new_by_cycle[cycle_id] = sorted(new_ids)
        active_by_cycle[cycle_id] = sorted(active_ids)
        exited_by_cycle[cycle_id] = sorted(exited_ids)
        cumulative_count_by_cycle[cycle_id] = len(seen_comment_ids)
        completed_cycle_ids.append(cycle_id)

        monitoring_inputs.append(
            _build_monitoring_input(
                cycle=cycle,
                active_ids=active_ids,
                new_ids=new_ids,
                exited_ids=exited_ids,
                cumulative_ids=seen_comment_ids,
                active_video_count=active_video_count,
            )
        )
        detection_inputs.append(
            _build_detection_input(
                cycle=cycle,
                active_ids=active_ids,
                new_ids=new_ids,
                exited_ids=exited_ids,
                cumulative_ids=seen_comment_ids,
                overlap_ids=overlap_ids,
            )
        )
        readiness_rows.append(
            _build_readiness_row(
                cycle=cycle,
                active_ids=active_ids,
                new_ids=new_ids,
                exited_ids=exited_ids,
                temporal=temporal,
                active_count_matches_processed=len(active_ids) == len(active_df),
                new_ids_unique_so_far=not duplicate_new_ids,
            )
        )
        previous_active_ids = active_ids

    prepared_refs = {
        "cycle_monitoring_inputs": "cycle_monitoring_inputs.jsonl",
        "cycle_detection_inputs": "cycle_detection_inputs.jsonl",
        "cycle_window_inventory": "cycle_window_inventory.csv",
        "cycle_detection_readiness_report": "cycle_detection_readiness_report.jsonl",
    }
    stateful_context = {
        "simulation_run_id": simulation_run_id,
        "simulation_mode": "cyclic_ingestion_simulation",
        "adapter_stage": "C-3",
        "adapter_mode": "stateful",
        "run_monitoring": False,
        "run_detection": False,
        "run_rag": False,
        "seen_comment_ids": sorted(seen_comment_ids),
        "seen_comment_count": len(seen_comment_ids),
        "new_comment_ids_by_cycle": new_by_cycle,
        "active_window_comment_ids_by_cycle": active_by_cycle,
        "exited_window_comment_ids_by_cycle": exited_by_cycle,
        "cumulative_comment_count_by_cycle": cumulative_count_by_cycle,
        "completed_cycle_ids": completed_cycle_ids,
        "prepared_signal_input_refs": prepared_refs,
        "emitted_event_registry_stub": {
            "status": "reserved_not_used_c3",
            "emitted_event_ids": [],
            "deduplication_policy": (
                "future detector stages must check emitted events, trigger_time, "
                "video_id, active_comment_ids_hash, and cooldown state before emitting"
            ),
        },
        "decision_state_stub": {
            "status": "reserved_not_used_c3",
            "stateful_decision_mode": True,
            "detector_state_policy": "stateful_required_future_stage",
        },
        "cooldown_state_stub": {
            "status": "reserved_not_used_c3",
            "cooldown_tracking_enabled_future_stage": True,
            "cooldown_entries": [],
        },
        "source_cycle_state": cycle_state,
    }
    readiness_failed = sum(
        1 for row in readiness_rows if row["readiness_status"] != "ready_for_future_monitoring_detection"
    )
    manifest = {
        "simulation_run_id": simulation_run_id,
        "simulation_mode": "cyclic_ingestion_simulation",
        "adapter_stage": "C-3",
        "adapter_mode": "stateful",
        "adapter_status": "prepared" if readiness_failed == 0 else "prepared_with_warnings",
        "interval_policy": simulation_manifest.get("interval_policy"),
        "temporal_rules": simulation_manifest.get("temporal_policy"),
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
            "total": len(cycles),
            "completed_stateful_preparation": len(completed_cycle_ids),
            "readiness_failed": readiness_failed,
            "cycles_with_window_overlap": overlap_cycle_count,
        },
        "comment_counts": {
            "seen_comment_count": len(seen_comment_ids),
            "window_inventory_rows": len(window_rows),
            "exited_window_memberships": exited_total,
            "active_window_memberships": sum(len(ids) for ids in active_by_cycle.values()),
        },
        "output_artifacts": prepared_refs
        | {
            "cycle_stateful_context": "cycle_stateful_context.json",
            "cycle_adapter_manifest": "cycle_adapter_manifest.json",
        },
        "compatibility": {
            "retrospective_replay_status": "untouched",
            "monitoring_status": "not_executed_c3",
            "detection_status": "not_executed_c3",
            "rag_status": "not_executed_c3",
        },
    }

    window_df = pd.DataFrame(window_rows)
    _write_csv(simulation_dir / "cycle_window_inventory.csv", window_df)
    _write_jsonl(simulation_dir / "cycle_monitoring_inputs.jsonl", monitoring_inputs)
    _write_jsonl(simulation_dir / "cycle_detection_inputs.jsonl", detection_inputs)
    _write_jsonl(
        simulation_dir / "cycle_detection_readiness_report.jsonl",
        readiness_rows,
    )
    _write_json(simulation_dir / "cycle_stateful_context.json", stateful_context)
    _write_json(simulation_dir / "cycle_adapter_manifest.json", manifest)

    return {
        "simulation_run_id": simulation_run_id,
        "simulation_dir": str(simulation_dir),
        "adapter_status": manifest["adapter_status"],
        "cycles_total": len(cycles),
        "readiness_failed": readiness_failed,
        "seen_comment_count": len(seen_comment_ids),
        "active_window_memberships": manifest["comment_counts"][
            "active_window_memberships"
        ],
        "exited_window_memberships": exited_total,
        "cycles_with_window_overlap": overlap_cycle_count,
        "artifacts": {
            "cycle_monitoring_inputs": str(simulation_dir / "cycle_monitoring_inputs.jsonl"),
            "cycle_detection_inputs": str(simulation_dir / "cycle_detection_inputs.jsonl"),
            "cycle_window_inventory": str(simulation_dir / "cycle_window_inventory.csv"),
            "cycle_stateful_context": str(simulation_dir / "cycle_stateful_context.json"),
            "cycle_detection_readiness_report": str(
                simulation_dir / "cycle_detection_readiness_report.jsonl"
            ),
            "cycle_adapter_manifest": str(simulation_dir / "cycle_adapter_manifest.json"),
        },
    }


__all__ = [
    "CyclicStatefulAdapterConfig",
    "run_cyclic_stateful_adapter",
]
