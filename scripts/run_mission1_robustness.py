#!/usr/bin/env python3
"""Mission 1 robustness experiment runner for mock perception/control noise."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages"))
sys.path.insert(0, str(ROOT / "sim_interfaces" / "airsim_zenoh_bridge"))

from mission_framework.config import MissionConfig  # noqa: E402

NOISE_LEVELS = {
    "0": 0.0,
    "low": 0.05,
    "medium": 0.10,
    "high": 0.20,
    "very_high": 0.40,
}

DEFAULT_START_POSES = [
    {"x": 0.0, "y": 0.0, "z": -5.0, "yaw_deg": 0.0},
    {"x": 2.5, "y": -1.5, "z": -5.5, "yaw_deg": 20.0},
    {"x": -3.0, "y": 1.0, "z": -4.5, "yaw_deg": -25.0},
]

TRAJECTORIES = ["straight", "curve", "speed_ramp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mission 1 robustness experiments")
    parser.add_argument(
        "--config",
        default=str(ROOT / "packages" / "mission_framework" / "configs" / "solitary_sentinel_tuned.yaml"),
        help="Mission config YAML to use as the experiment baseline",
    )
    parser.add_argument("--connect", default="tcp/127.0.0.1:7447", help="Zenoh endpoint")
    parser.add_argument("--python", default=sys.executable, help="Python executable for child processes")
    parser.add_argument("--repeats", type=int, default=3, help="Runs per noise pair")
    parser.add_argument(
        "--sensor-levels",
        default=",".join(NOISE_LEVELS.keys()),
        help="Comma-separated sensor noise levels to evaluate",
    )
    parser.add_argument(
        "--actuator-levels",
        default=",".join(NOISE_LEVELS.keys()),
        help="Comma-separated actuator noise levels to evaluate",
    )
    parser.add_argument(
        "--trajectories",
        default=",".join(TRAJECTORIES),
        help="Comma-separated trajectory profiles to evaluate",
    )
    parser.add_argument("--latency-ms", type=float, default=150.0, help="Fixed detection latency in milliseconds")
    parser.add_argument("--dropout-burst-prob", type=float, default=0.02, help="Probability of starting a detection dropout burst")
    parser.add_argument("--dropout-burst-min-s", type=float, default=0.5, help="Minimum dropout burst duration in seconds")
    parser.add_argument("--dropout-burst-max-s", type=float, default=1.5, help="Maximum dropout burst duration in seconds")
    parser.add_argument(
        "--output-dir",
        default=str(ROOT / "outputs" / "mission1_robustness"),
        help="Output directory for experiment artifacts",
    )
    parser.add_argument("--bridge-start-delay", type=float, default=3.0, help="Seconds to wait after bridge startup")
    parser.add_argument("--detector-start-delay", type=float, default=2.0, help="Seconds to wait after detector startup")
    parser.add_argument("--mission-timeout", type=float, default=120.0, help="Timeout for a single mission run in seconds")
    return parser.parse_args()


def acceptance_pass(metrics: dict) -> bool:
    acceptance = metrics["acceptance"]
    return (
        metrics["track_ratio"] >= acceptance["min_track_ratio"]
        and metrics["avg_abs_bearing_x"] <= acceptance["max_avg_abs_bearing_x"]
        and metrics["avg_abs_area_error_ratio"] <= acceptance["max_avg_abs_area_error_ratio"]
        and metrics["avg_command_delta"] <= acceptance["max_avg_command_delta"]
    )


def metrics_output_path(config_path: Path) -> Path:
    config = MissionConfig.from_yaml(config_path)
    return ROOT / config.output_dir / f"{config.name}_metrics.json"


def start_process(cmd: list[str], log_path: Path, env: dict[str, str], cwd: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    proc._log_handle = log_handle  # type: ignore[attr-defined]
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None:
        return
    if proc.poll() is None:
        try:
            if os.name == "nt":
                proc.send_signal(signal.CTRL_BREAK_EVENT)
                time.sleep(1.0)
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    log_handle = getattr(proc, "_log_handle", None)
    if log_handle is not None:
        log_handle.close()


def run_one(
    python_exe: str,
    connect: str,
    config_path: Path,
    output_dir: Path,
    sensor_level: str,
    actuator_level: str,
    repeat_idx: int,
    pose: dict,
    trajectory: str,
    latency_ms: float,
    dropout_burst_prob: float,
    dropout_burst_min_s: float,
    dropout_burst_max_s: float,
    bridge_delay: float,
    detector_delay: float,
    mission_timeout: float,
) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'packages'};{ROOT / 'sim_interfaces' / 'airsim_zenoh_bridge'}"

    run_name = (
        f"trajectory_{trajectory}__sensor_{sensor_level}__actuator_{actuator_level}__run_{repeat_idx + 1}"
    )
    run_dir = output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_output_path(config_path)
    if metrics_path.exists():
        metrics_path.unlink()

    bridge_cmd = [
        python_exe,
        "-u",
        "sim_interfaces/airsim_zenoh_bridge/mock_bridge.py",
        "--zenoh-listen",
        "tcp/0.0.0.0:7447",
        "--actuator-noise-level",
        actuator_level,
        "--start-x",
        str(pose["x"]),
        "--start-y",
        str(pose["y"]),
        "--start-z",
        str(pose["z"]),
        "--start-yaw-deg",
        str(pose["yaw_deg"]),
        "--seed",
        str(10_000 + repeat_idx),
    ]
    detector_cmd = [
        python_exe,
        "-u",
        "sim_interfaces/airsim_zenoh_bridge/mock_detection_publisher.py",
        "--connect",
        connect,
        "--camera",
        "front",
        "--target-class",
        "orange ball",
        "--sensor-noise-level",
        sensor_level,
        "--trajectory",
        trajectory,
        "--latency-ms",
        str(latency_ms),
        "--dropout-burst-prob",
        str(dropout_burst_prob),
        "--dropout-burst-min-s",
        str(dropout_burst_min_s),
        "--dropout-burst-max-s",
        str(dropout_burst_max_s),
        "--seed",
        str(20_000 + repeat_idx),
    ]
    mission_cmd = [
        python_exe,
        "sim_interfaces/airsim_zenoh_bridge/run_mission.py",
        "--connect",
        connect,
        "--config",
        str(config_path),
    ]

    bridge_proc = detector_proc = None
    mission_log = run_dir / "mission.log"
    mission_returncode = None
    started_at = time.time()
    try:
        bridge_proc = start_process(bridge_cmd, run_dir / "bridge.log", env, ROOT)
        time.sleep(bridge_delay)
        detector_proc = start_process(detector_cmd, run_dir / "detector.log", env, ROOT)
        time.sleep(detector_delay)

        with open(mission_log, "w", encoding="utf-8") as log_handle:
            mission_proc = subprocess.run(
                mission_cmd,
                cwd=ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                timeout=mission_timeout,
            )
            mission_returncode = mission_proc.returncode
    finally:
        stop_process(detector_proc)
        stop_process(bridge_proc)

    elapsed = time.time() - started_at

    if not metrics_path.exists():
        return {
            "trajectory": trajectory,
            "sensor_noise": sensor_level,
            "actuator_noise": actuator_level,
            "repeat": repeat_idx + 1,
            "start_pose": pose,
            "mission_returncode": mission_returncode,
            "elapsed_s": elapsed,
            "metrics_found": False,
            "pass": False,
        }

    run_metrics_path = run_dir / "metrics.json"
    shutil.copy2(metrics_path, run_metrics_path)
    with open(run_metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    return {
        "trajectory": trajectory,
        "sensor_noise": sensor_level,
        "actuator_noise": actuator_level,
        "repeat": repeat_idx + 1,
        "start_pose": pose,
        "mission_returncode": mission_returncode,
        "elapsed_s": elapsed,
        "metrics_found": True,
        "pass": acceptance_pass(metrics) and mission_returncode == 0,
        "metrics": metrics,
    }


def aggregate_results(results: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, str], list[dict]] = {}
    for item in results:
        grouped.setdefault((item["trajectory"], item["sensor_noise"], item["actuator_noise"]), []).append(item)

    aggregates = []
    for (trajectory, sensor_level, actuator_level), runs in grouped.items():
        metric_runs = [r["metrics"] for r in runs if r.get("metrics_found")]
        pass_count = sum(1 for r in runs if r["pass"])
        aggregate = {
            "trajectory": trajectory,
            "sensor_noise": sensor_level,
            "actuator_noise": actuator_level,
            "runs": len(runs),
            "pass_count": pass_count,
            "pass_rate": pass_count / len(runs) if runs else 0.0,
            "avg_track_ratio": mean(m["track_ratio"] for m in metric_runs) if metric_runs else 0.0,
            "avg_abs_bearing_x": mean(m["avg_abs_bearing_x"] for m in metric_runs) if metric_runs else 0.0,
            "avg_abs_area_error_ratio": mean(m["avg_abs_area_error_ratio"] for m in metric_runs) if metric_runs else 0.0,
            "avg_command_delta": mean(m["avg_command_delta"] for m in metric_runs) if metric_runs else 0.0,
        }
        aggregates.append(aggregate)
    return sorted(
        aggregates,
        key=lambda x: (
            TRAJECTORIES.index(x["trajectory"]),
            list(NOISE_LEVELS).index(x["sensor_noise"]),
            list(NOISE_LEVELS).index(x["actuator_noise"]),
        ),
    )


def stable_summary(aggregates: list[dict]) -> dict:
    fully_stable = [a for a in aggregates if a["pass_rate"] == 1.0]
    failures = [a for a in aggregates if a["pass_rate"] < 1.0]

    def noise_rank(name: str) -> float:
        return NOISE_LEVELS[name]

    max_sensor_stable = max((a["sensor_noise"] for a in fully_stable), key=noise_rank, default="0")
    max_actuator_stable = max((a["actuator_noise"] for a in fully_stable), key=noise_rank, default="0")

    return {
        "fully_stable_pairs": len(fully_stable),
        "max_sensor_noise_with_full_pass": max_sensor_stable,
        "max_actuator_noise_with_full_pass": max_actuator_stable,
        "failure_conditions": [
            {
                "trajectory": a["trajectory"],
                "sensor_noise": a["sensor_noise"],
                "actuator_noise": a["actuator_noise"],
                "pass_rate": a["pass_rate"],
            }
            for a in failures
        ],
    }


def write_markdown_report(output_dir: Path, summary_json: dict) -> Path:
    summary = summary_json["summary"]
    aggregates = summary_json["aggregates"]
    failures = summary["failure_conditions"]
    stable_pairs = [a for a in aggregates if a["pass_rate"] == 1.0]

    report_lines = [
        "# Mission 1 Robustness Report",
        "",
        f"Config: `{summary_json['config']}`",
        "",
        "## Scenario Setup",
        "",
        f"- Trajectories: {', '.join(summary_json['trajectories'])}",
        f"- Sensor noise levels: {', '.join(summary_json['sensor_noise_levels'])}",
        f"- Actuator noise levels: {', '.join(summary_json['actuator_noise_levels'])}",
        f"- Repeats per noise pair: {summary_json['repeats']}",
        f"- Detection latency: {summary_json['latency_ms']} ms",
        f"- Dropout burst probability: {summary_json['dropout_burst_prob']}",
        f"- Dropout burst window: {summary_json['dropout_burst_min_s']}-{summary_json['dropout_burst_max_s']} s",
        "",
        "## Top-Level Result",
        "",
        f"- Fully stable trajectory/noise pairs: {summary['fully_stable_pairs']}",
        f"- Stable up to sensor noise: `{summary['max_sensor_noise_with_full_pass']}`",
        f"- Stable up to actuator noise: `{summary['max_actuator_noise_with_full_pass']}`",
        "",
        "## Stable Conditions",
        "",
    ]

    for item in stable_pairs[:15]:
        report_lines.append(
            f"- `{item['trajectory']}` | sensor=`{item['sensor_noise']}` | actuator=`{item['actuator_noise']}` | pass_rate={item['pass_rate']:.2f}"
        )

    report_lines.extend(["", "## Failure Conditions", ""])
    if failures:
        for item in failures:
            report_lines.append(
                f"- `{item['trajectory']}` | sensor=`{item['sensor_noise']}` | actuator=`{item['actuator_noise']}` | pass_rate={item['pass_rate']:.2f}"
            )
    else:
        report_lines.append("- No failure conditions observed.")

    report_lines.extend([
        "",
        "## Notes",
        "",
        "- Temporal dropout bursts and fixed detection latency were enabled to make the mock perception path more realistic.",
        "- Multiple target trajectories were exercised instead of a single oscillating path.",
        "- Use the JSON/CSV outputs for deeper analysis and the per-run folders for raw logs.",
        "",
    ])

    report_path = output_dir / "summary_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    return report_path


def main() -> int:
    args = parse_args()
    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sensor_levels = [x.strip() for x in args.sensor_levels.split(",") if x.strip()]
    actuator_levels = [x.strip() for x in args.actuator_levels.split(",") if x.strip()]
    trajectories = [x.strip() for x in args.trajectories.split(",") if x.strip()]

    for level in sensor_levels + actuator_levels:
        if level not in NOISE_LEVELS:
            raise SystemExit(f"Unknown noise level: {level}")
    for trajectory in trajectories:
        if trajectory not in TRAJECTORIES:
            raise SystemExit(f"Unknown trajectory: {trajectory}")

    results = []
    for trajectory in trajectories:
        for sensor_level in sensor_levels:
            for actuator_level in actuator_levels:
                for repeat_idx in range(args.repeats):
                    pose = DEFAULT_START_POSES[repeat_idx % len(DEFAULT_START_POSES)]
                    print(
                        f"Running trajectory={trajectory} sensor={sensor_level} actuator={actuator_level} "
                        f"repeat={repeat_idx + 1}/{args.repeats} start=({pose['x']}, {pose['y']}, {pose['z']}, {pose['yaw_deg']})"
                    )
                    result = run_one(
                        python_exe=args.python,
                        connect=args.connect,
                        config_path=config_path,
                        output_dir=output_dir,
                        sensor_level=sensor_level,
                        actuator_level=actuator_level,
                        repeat_idx=repeat_idx,
                        pose=pose,
                        trajectory=trajectory,
                        latency_ms=args.latency_ms,
                        dropout_burst_prob=args.dropout_burst_prob,
                        dropout_burst_min_s=args.dropout_burst_min_s,
                        dropout_burst_max_s=args.dropout_burst_max_s,
                        bridge_delay=args.bridge_start_delay,
                        detector_delay=args.detector_start_delay,
                        mission_timeout=args.mission_timeout,
                    )
                    results.append(result)
                    status = "PASS" if result["pass"] else "FAIL"
                    print(f"  -> {status}")

    aggregates = aggregate_results(results)
    summary = stable_summary(aggregates)

    summary_json = {
        "config": str(config_path),
        "trajectories": trajectories,
        "sensor_noise_levels": sensor_levels,
        "actuator_noise_levels": actuator_levels,
        "repeats": args.repeats,
        "start_poses": DEFAULT_START_POSES,
        "latency_ms": args.latency_ms,
        "dropout_burst_prob": args.dropout_burst_prob,
        "dropout_burst_min_s": args.dropout_burst_min_s,
        "dropout_burst_max_s": args.dropout_burst_max_s,
        "runs": results,
        "aggregates": aggregates,
        "summary": summary,
    }

    json_path = output_dir / "summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2)

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "trajectory",
                "sensor_noise",
                "actuator_noise",
                "runs",
                "pass_count",
                "pass_rate",
                "avg_track_ratio",
                "avg_abs_bearing_x",
                "avg_abs_area_error_ratio",
                "avg_command_delta",
            ],
        )
        writer.writeheader()
        writer.writerows(aggregates)

    report_path = write_markdown_report(output_dir, summary_json)

    print()
    print(f"Summary JSON:   {json_path}")
    print(f"Summary CSV:    {csv_path}")
    print(f"Summary report: {report_path}")
    print(
        f"System stable up to sensor noise '{summary['max_sensor_noise_with_full_pass']}' "
        f"and actuator noise '{summary['max_actuator_noise_with_full_pass']}' (among fully passing pairs)."
    )
    if summary["failure_conditions"]:
        print("Fails under:")
        for item in summary["failure_conditions"][:20]:
            print(
                f"  trajectory={item['trajectory']} sensor={item['sensor_noise']} "
                f"actuator={item['actuator_noise']} pass_rate={item['pass_rate']:.2f}"
            )
    else:
        print("No failure conditions observed in the evaluated grid.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
