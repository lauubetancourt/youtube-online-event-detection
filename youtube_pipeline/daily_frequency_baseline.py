from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


DETECTOR_NAME = "daily_frequency_baseline"
DEFAULT_COOLDOWN_POLICY = "disabled_for_daily_detection"
CONFIGURED_COOLDOWN_POLICY = "configured_cooldown"
VALID_COOLDOWN_POLICIES = {DEFAULT_COOLDOWN_POLICY, CONFIGURED_COOLDOWN_POLICY}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON artifact not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_ready(row), ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return _format_utc(value)
    if isinstance(value, float) and pd.isna(value):
        return None
    return value


def _format_utc(value: Any) -> str | None:
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(ts):
        return None
    return pd.Timestamp(ts).strftime("%Y-%m-%dT%H:%M:%SZ")


def _now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha1_short(value: str, length: int = 12) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()[:length]


@dataclass
class DailyFrequencyBaselineConfig:
    simulation_dir: str | Path
    output_dir: str | Path | None = None
    signal_name: str = "new_comment_count"
    baseline_window_size_cycles: int = 3
    k_multiplier: float = 2.0
    min_count: int = 500
    min_delta: float = 250.0
    min_pct_change: float = 0.5
    warmup_cycles: int = 3
    cooldown_cycles: int = 0
    cooldown_policy: str = DEFAULT_COOLDOWN_POLICY
    use_pct_change: bool = True
    use_delta: bool = True
    trigger_on_increase_only: bool = True
    timezone: str = "America/Bogota"
    canonical_timezone: str = "UTC"
    parameter_status: str = "exploratory_defaults"
    run_rag: bool = False
    run_llm: bool = False
    run_serper: bool = False
    use_embeddings: bool = False
    use_vectorstore: bool = False

    def __post_init__(self) -> None:
        if self.cooldown_cycles > 0 and self.cooldown_policy == DEFAULT_COOLDOWN_POLICY:
            self.cooldown_policy = CONFIGURED_COOLDOWN_POLICY

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DailyFrequencyBaselineConfig":
        config_payload = payload.get("daily_frequency_baseline", payload)
        if not isinstance(config_payload, dict):
            raise ValueError("Daily frequency baseline config must be an object.")
        allowed = set(cls.__dataclass_fields__)
        unknown = sorted(set(config_payload) - allowed)
        if unknown:
            raise ValueError(f"Unknown daily frequency baseline config fields: {unknown}")
        return cls(**config_payload)

    def simulation_path(self) -> Path:
        return Path(self.simulation_dir)

    def output_path(self) -> Path:
        return Path(self.output_dir) if self.output_dir is not None else self.simulation_path()

    def validate(self) -> None:
        if self.baseline_window_size_cycles < 1:
            raise ValueError("baseline_window_size_cycles must be >= 1.")
        if self.k_multiplier <= 0:
            raise ValueError("k_multiplier must be > 0.")
        if self.min_count < 0:
            raise ValueError("min_count must be >= 0.")
        if self.min_delta < 0:
            raise ValueError("min_delta must be >= 0.")
        if self.min_pct_change < 0:
            raise ValueError("min_pct_change must be >= 0.")
        if self.warmup_cycles < 0:
            raise ValueError("warmup_cycles must be >= 0.")
        if self.cooldown_cycles < 0:
            raise ValueError("cooldown_cycles must be >= 0.")
        if self.cooldown_policy not in VALID_COOLDOWN_POLICIES:
            raise ValueError(
                "cooldown_policy must be one of: "
                + ", ".join(sorted(VALID_COOLDOWN_POLICIES))
            )
        if self.cooldown_cycles == 0 and self.cooldown_policy != DEFAULT_COOLDOWN_POLICY:
            raise ValueError(
                "cooldown_policy must be disabled_for_daily_detection when cooldown_cycles is 0."
            )
        if self.canonical_timezone != "UTC":
            raise ValueError("canonical_timezone must remain UTC.")
        forbidden = {
            "run_rag": self.run_rag,
            "run_llm": self.run_llm,
            "run_serper": self.run_serper,
            "use_embeddings": self.use_embeddings,
            "use_vectorstore": self.use_vectorstore,
        }
        enabled = [name for name, value in forbidden.items() if value]
        if enabled:
            raise ValueError(
                f"{DETECTOR_NAME} is local only. These flags must remain false: "
                + ", ".join(enabled)
            )


def _validate_inputs(
    *,
    config: DailyFrequencyBaselineConfig,
    signal_manifest: dict[str, Any],
    signal_rows: list[dict[str, Any]],
    quality_rows: list[dict[str, Any]],
) -> list[str]:
    errors: list[str] = []
    if signal_manifest.get("stage") != "C-5":
        errors.append("cycle_signal_manifest.json must come from C-5.")
    if signal_manifest.get("mode") != "signals_dry_run":
        errors.append("cycle_signal_manifest.json must use mode='signals_dry_run'.")
    guards = signal_manifest.get("execution_guards", {})
    for key in ["run_detection", "run_rag"]:
        if guards.get(key) is not False:
            errors.append(f"C-5 execution guard {key} must be false.")
    if not signal_rows:
        errors.append("cycle_signal_series.jsonl must contain at least one row.")
    required = {
        "simulation_run_id",
        "cycle_id",
        "cycle_index",
        "signal_date",
        "observation_time_utc",
        "analysis_window_start_utc",
        "analysis_window_end_utc",
        "data_cutoff_utc",
        config.signal_name,
        "active_window_comment_count",
        "delta_active_window_comment_count",
        "pct_change_active_window_comment_count",
        "active_video_count",
        "comment_ids_hash",
        "join_status",
        "temporal_status",
        "schema_status",
    }
    missing = sorted(required - set(signal_rows[0]))
    if missing:
        errors.append(f"cycle_signal_series.jsonl missing fields: {missing}")
    if quality_rows and len(quality_rows) != len(signal_rows):
        errors.append("cycle_signal_quality_report.jsonl row count must match signal series.")
    return errors


def _ordered_signal_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["cycle_index"]),
            pd.Timestamp(row["observation_time_utc"]),
            str(row["cycle_id"]),
        ),
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    result = pd.to_numeric(value, errors="coerce")
    if pd.isna(result):
        return None
    return float(result)


def _pct_change(current_value: float, previous_value: float | None) -> tuple[float | None, str]:
    if previous_value is None:
        return None, "undefined_no_previous_value"
    if previous_value == 0:
        return None, "undefined_previous_zero"
    return float((current_value - previous_value) / previous_value), "defined"


def _daily_event_id(
    *,
    detector_name: str,
    cycle_id: str,
    signal_name: str,
    signal_value: float,
) -> str:
    return "dfe_" + _sha1_short(
        "|".join([detector_name, str(cycle_id), signal_name, str(signal_value)])
    )


def _build_score_and_quality_rows(
    *,
    config: DailyFrequencyBaselineConfig,
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    scores: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    cooldown_remaining = 0
    created_at_utc = _now_utc()

    for index, row in enumerate(rows):
        cycle_id = str(row["cycle_id"])
        current_value = _safe_float(row.get(config.signal_name))
        previous_value = _safe_float(rows[index - 1].get(config.signal_name)) if index > 0 else None
        delta_value = None if current_value is None or previous_value is None else current_value - previous_value
        pct_change_value, pct_change_status = (
            (None, "undefined_current_null")
            if current_value is None
            else _pct_change(current_value, previous_value)
        )
        baseline_rows = rows[max(0, index - config.baseline_window_size_cycles) : index]
        baseline_values = [_safe_float(item.get(config.signal_name)) for item in baseline_rows]
        baseline_values = [value for value in baseline_values if value is not None]
        baseline_available = len(baseline_values) == config.baseline_window_size_cycles
        baseline_mean = float(sum(baseline_values) / len(baseline_values)) if baseline_available else None
        threshold_value = (
            float(config.k_multiplier * baseline_mean)
            if baseline_mean is not None
            else None
        )
        ratio_to_baseline = (
            None
            if current_value is None or baseline_mean in {None, 0}
            else float(current_value / baseline_mean)
        )

        warmup_status = (
            "warmup"
            if index < config.warmup_cycles or not baseline_available
            else "ready"
        )
        cooldown_status = "active" if cooldown_remaining > 0 else "inactive"
        condition_results = {
            "has_signal_value": current_value is not None,
            "baseline_available": baseline_available,
            "min_count": current_value is not None and current_value >= config.min_count,
            "baseline_threshold": (
                current_value is not None
                and threshold_value is not None
                and current_value > threshold_value
            ),
            "min_delta": (
                True
                if not config.use_delta
                else delta_value is not None and delta_value >= config.min_delta
            ),
            "increase_only": (
                True
                if not config.trigger_on_increase_only
                else delta_value is not None and delta_value > 0
            ),
            "min_pct_change": (
                True
                if not config.use_pct_change
                else pct_change_value is not None
                and pct_change_value >= config.min_pct_change
            ),
            "warmup_complete": warmup_status == "ready",
            "cooldown_clear": cooldown_status == "inactive",
        }
        trigger_candidate = all(condition_results.values())
        if trigger_candidate:
            trigger_reason = (
                f"{config.signal_name}={current_value:g} > "
                f"{config.k_multiplier:g}*baseline_mean={threshold_value:g}; "
                f"delta={delta_value:g}; pct_change={pct_change_value:g}"
            )
            event = {
                "daily_event_id": _daily_event_id(
                    detector_name=DETECTOR_NAME,
                    cycle_id=cycle_id,
                    signal_name=config.signal_name,
                    signal_value=current_value,
                ),
                "cycle_id": cycle_id,
                "cycle_index": int(row["cycle_index"]),
                "signal_name": config.signal_name,
                "signal_value": current_value,
                "baseline_mean": baseline_mean,
                "baseline_window_cycle_ids": [item["cycle_id"] for item in baseline_rows],
                "ratio_to_baseline": ratio_to_baseline,
                "delta_value": delta_value,
                "pct_change_value": pct_change_value,
                "pct_change_status": pct_change_status,
                "threshold_value": threshold_value,
                "trigger_reason": trigger_reason,
                "analysis_window_start_utc": row["analysis_window_start_utc"],
                "analysis_window_end_utc": row["analysis_window_end_utc"],
                "data_cutoff_utc": row["data_cutoff_utc"],
                "support_comment_count": int(row["active_window_comment_count"]),
                "active_video_count": int(row["active_video_count"]),
                "comment_ids_hash": row["comment_ids_hash"],
                "detector_name": DETECTOR_NAME,
                "detector_config": asdict(config),
                "created_at_utc": created_at_utc,
            }
            events.append(event)
            cooldown_remaining = config.cooldown_cycles
        else:
            failed = [name for name, passed in condition_results.items() if not passed]
            trigger_reason = "not_triggered:" + ",".join(failed)

        score = {
            "cycle_id": cycle_id,
            "cycle_index": int(row["cycle_index"]),
            "signal_date": row["signal_date"],
            "observation_time_utc": row["observation_time_utc"],
            "signal_name": config.signal_name,
            "signal_value": current_value,
            "baseline_mean": baseline_mean,
            "baseline_window_cycle_ids": [item["cycle_id"] for item in baseline_rows],
            "baseline_window_size_cycles": config.baseline_window_size_cycles,
            "ratio_to_baseline": ratio_to_baseline,
            "delta_value": delta_value,
            "pct_change_value": pct_change_value,
            "pct_change_status": pct_change_status,
            "threshold_value": threshold_value,
            "warmup_status": warmup_status,
            "cooldown_status": cooldown_status,
            "condition_results": condition_results,
            "trigger_candidate": trigger_candidate,
            "trigger_reason": trigger_reason,
            "score_status": "scored",
            "analysis_window_start_utc": row["analysis_window_start_utc"],
            "analysis_window_end_utc": row["analysis_window_end_utc"],
            "data_cutoff_utc": row["data_cutoff_utc"],
            "support_comment_count": int(row["active_window_comment_count"]),
            "active_video_count": int(row["active_video_count"]),
            "comment_ids_hash": row["comment_ids_hash"],
        }
        scores.append(score)
        quality.append(
            {
                "cycle_id": cycle_id,
                "cycle_index": int(row["cycle_index"]),
                "quality_status": "passed",
                "signal_name": config.signal_name,
                "signal_missing": config.signal_name not in row,
                "signal_null": current_value is None,
                "baseline_available": baseline_available,
                "warmup_status": warmup_status,
                "pct_change_status": pct_change_status,
                "join_status": row.get("join_status"),
                "temporal_status": row.get("temporal_status"),
                "schema_status": row.get("schema_status"),
                "rag_execution_status": "not_executed",
                "external_calls": {
                    "llm": 0,
                    "serper": 0,
                    "embeddings": False,
                    "vectorstore": False,
                },
            }
        )
        if cooldown_remaining > 0 and trigger_candidate:
            # The current cycle consumes the trigger; subsequent cycles observe cooldown.
            pass
        elif cooldown_remaining > 0:
            cooldown_remaining -= 1

    return scores, events, quality


def run_daily_frequency_baseline(config: DailyFrequencyBaselineConfig) -> dict[str, Any]:
    config.validate()
    simulation_dir = config.simulation_path()
    output_dir = config.output_path()
    signal_manifest = _read_json(simulation_dir / "cycle_signal_manifest.json")
    signal_rows = _read_jsonl(simulation_dir / "cycle_signal_series.jsonl")
    signal_quality_rows = _read_jsonl(simulation_dir / "cycle_signal_quality_report.jsonl")
    input_errors = _validate_inputs(
        config=config,
        signal_manifest=signal_manifest,
        signal_rows=signal_rows,
        quality_rows=signal_quality_rows,
    )
    if input_errors:
        raise ValueError("Daily frequency baseline input validation failed: " + "; ".join(input_errors))

    ordered_rows = _ordered_signal_rows(signal_rows)
    scores, events, quality_rows = _build_score_and_quality_rows(
        config=config,
        rows=ordered_rows,
    )
    warmup_count = sum(1 for row in scores if row["warmup_status"] == "warmup")
    cooldown_count = sum(1 for row in scores if row["cooldown_status"] == "active")
    manifest = {
        "detector_name": DETECTOR_NAME,
        "detector_kind": "daily_frequency_baseline_external_to_xiao",
        "status": "completed",
        "parameter_status": config.parameter_status,
        "detector_config": asdict(config),
        "input_artifacts": {
            "cycle_signal_series": str(simulation_dir / "cycle_signal_series.jsonl"),
            "cycle_signal_manifest": str(simulation_dir / "cycle_signal_manifest.json"),
            "cycle_signal_quality_report": str(
                simulation_dir / "cycle_signal_quality_report.jsonl"
            ),
        },
        "output_artifacts": {
            "cycle_daily_frequency_scores": "cycle_daily_frequency_scores.jsonl",
            "cycle_daily_frequency_events": "cycle_daily_frequency_events.jsonl",
            "cycle_daily_frequency_detector_manifest": (
                "cycle_daily_frequency_detector_manifest.json"
            ),
            "cycle_daily_frequency_quality_report": (
                "cycle_daily_frequency_quality_report.jsonl"
            ),
        },
        "run_summary": {
            "cycles_evaluated": len(scores),
            "warmup_cycles": warmup_count,
            "evaluable_cycles": len(scores) - warmup_count,
            "cooldown_cycles": cooldown_count,
            "cooldown_policy": config.cooldown_policy,
            "events_detected": len(events),
        },
        "compatibility": {
            "xiao_status": "untouched_not_executed",
            "retrospective_replay_status": "untouched",
            "rag_status": "not_executed",
            "sidecars_status": "untouched",
            "g1_g2_results_status": "untouched",
        },
        "execution_guards": {
            "run_rag": False,
            "llm_calls": 0,
            "serper_calls": 0,
            "embeddings": False,
            "vectorstore": False,
        },
        "created_at_utc": _now_utc(),
    }

    _write_jsonl(output_dir / "cycle_daily_frequency_scores.jsonl", scores)
    _write_jsonl(output_dir / "cycle_daily_frequency_events.jsonl", events)
    _write_json(output_dir / "cycle_daily_frequency_detector_manifest.json", manifest)
    _write_jsonl(output_dir / "cycle_daily_frequency_quality_report.jsonl", quality_rows)

    return {
        "detector_name": DETECTOR_NAME,
        "simulation_dir": str(simulation_dir),
        "output_dir": str(output_dir),
        "cycles_evaluated": len(scores),
        "warmup_cycles": warmup_count,
        "evaluable_cycles": len(scores) - warmup_count,
        "cooldown_cycles": cooldown_count,
        "cooldown_policy": config.cooldown_policy,
        "events_detected": len(events),
        "artifacts": {
            "cycle_daily_frequency_scores": str(
                output_dir / "cycle_daily_frequency_scores.jsonl"
            ),
            "cycle_daily_frequency_events": str(
                output_dir / "cycle_daily_frequency_events.jsonl"
            ),
            "cycle_daily_frequency_detector_manifest": str(
                output_dir / "cycle_daily_frequency_detector_manifest.json"
            ),
            "cycle_daily_frequency_quality_report": str(
                output_dir / "cycle_daily_frequency_quality_report.jsonl"
            ),
        },
    }


__all__ = [
    "CONFIGURED_COOLDOWN_POLICY",
    "DEFAULT_COOLDOWN_POLICY",
    "DETECTOR_NAME",
    "DailyFrequencyBaselineConfig",
    "run_daily_frequency_baseline",
]
