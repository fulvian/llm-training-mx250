#!/usr/bin/env python3
"""
Monitoraggio training in tempo reale con visualizzazione migliorata.

Usage:
    python3 monitor_training.py
"""

import json
import os
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.progress import Progress, BarColumn, TextColumn
    from rich import box
    from rich import markup
    from rich.style import Style
except ImportError:
    print("Installare rich: pip install rich")
    sys.exit(1)


COLORS = {
    "success": "bold green",
    "warning": "bold yellow",
    "error": "bold red",
    "info": "bold cyan",
    "metric": "bold magenta",
    "value": "bright_white",
    "progress_low": "red",
    "progress_mid": "yellow",
    "progress_high": "green",
    "header": "bold bright_cyan",
    "chart_train": "green",
    "chart_eval": "yellow",
    "vram_low": "green",
    "vram_mid": "yellow",
    "vram_high": "red",
    "temp_cool": "blue",
    "temp_warm": "green",
    "temp_hot": "yellow",
    "temp_critical": "red",
}

console = Console()


@dataclass
class LossTracker:
    loss_history: deque = field(default_factory=lambda: deque(maxlen=50))
    prev_loss: Optional[float] = None

    def add_loss(self, loss: float) -> None:
        self.prev_loss = self.loss_history[-1] if self.loss_history else None
        self.loss_history.append(loss)

    def get_trend(self) -> tuple[str, str]:
        if self.prev_loss is None or len(self.loss_history) < 2:
            return "→", COLORS["warning"]

        change = self.prev_loss - self.loss_history[-1]
        change_pct = abs(change / self.prev_loss) * 100 if self.prev_loss != 0 else 0

        if change_pct < 0.5:
            return "→", COLORS["warning"]
        elif change > 0:
            return "↓", COLORS["success"]
        else:
            return "↑", COLORS["error"]

    def get_change_rate(self) -> str:
        if len(self.loss_history) < 5:
            return "N/A"

        recent = list(self.loss_history)[-5:]
        if len(recent) < 2:
            return "N/A"

        total_change = (
            (recent[0] - recent[-1]) / recent[0] * 100 if recent[0] != 0 else 0
        )

        if total_change > 5:
            return f"[{COLORS['success']}]⚡ Fast ({total_change:.1f}%)[/{COLORS['success']}]"
        elif total_change > 1:
            return f"[{COLORS['warning']}]→ Steady ({total_change:.1f}%)[/{COLORS['warning']}]"
        elif total_change > -1:
            return f"[{COLORS['info']}]≈ Converging ({total_change:.1f}%)[/{COLORS['info']}]"
        else:
            return f"[{COLORS['error']}]↑ Increasing ({total_change:.1f}%)[/{COLORS['error']}]"


loss_tracker = LossTracker()


def format_decimal(value: Optional[float], precision: int = 6) -> str:
    """Format number in decimal notation with appropriate precision."""
    if value is None:
        return "N/A"
    if abs(value) >= 1000 or (abs(value) < 0.0001 and value != 0):
        return f"{value:.8f}"
    return f"{value:.{precision}f}"


def get_progress_color(pct: float) -> str:
    if pct < 30:
        return COLORS["progress_low"]
    elif pct < 70:
        return COLORS["progress_mid"]
    return COLORS["progress_high"]


def get_temp_color(temp: float) -> str:
    if temp < 60:
        return COLORS["temp_cool"]
    elif temp < 75:
        return COLORS["temp_warm"]
    elif temp < 85:
        return COLORS["temp_hot"]
    return COLORS["temp_critical"]


def get_vram_color(vram_pct: float) -> str:
    if vram_pct < 50:
        return COLORS["vram_low"]
    elif vram_pct < 80:
        return COLORS["vram_mid"]
    return COLORS["vram_high"]


def get_util_color(util: float) -> str:
    if util < 50:
        return COLORS["success"]
    elif util < 80:
        return COLORS["warning"]
    return COLORS["error"]


def draw_loss_chart(
    train_losses: list[float],
    eval_loss: Optional[float] = None,
    width: int = 40,
    height: int = 6,
) -> str:
    if not train_losses or len(train_losses) < 2:
        return "[dim]Raccogliendo dati...[/dim]"

    all_losses = train_losses.copy()
    if eval_loss is not None:
        all_losses.append(eval_loss)

    min_val = min(all_losses)
    max_val = max(all_losses)
    current_val = train_losses[-1]
    val_range = max_val - min_val if max_val != min_val else max_val * 0.1

    if val_range == 0:
        val_range = max_val * 0.1 if max_val != 0 else 0.001

    def get_y_pos(val: float) -> int:
        normalized = (val - min_val) / val_range
        return height - 1 - int(normalized * (height - 1))

    grid = [[" " for _ in range(width)] for _ in range(height)]

    step = max(1, len(train_losses) // width)
    sampled_losses = train_losses[::step][:width]

    prev_y = None
    for i, loss in enumerate(sampled_losses):
        y = get_y_pos(loss)
        y = max(0, min(height - 1, y))
        grid[y][i] = "●"

        if prev_y is not None and prev_y != y:
            direction = 1 if y > prev_y else -1
            for fill_y in range(prev_y + direction, y, direction):
                if 0 <= fill_y < height:
                    grid[fill_y][i] = "│"
        prev_y = y

    if eval_loss is not None and len(sampled_losses) > 0:
        eval_y = get_y_pos(eval_loss)
        eval_y = max(0, min(height - 1, eval_y))
        eval_x = min(len(sampled_losses) - 1, width - 2)
        grid[eval_y][eval_x] = "◆"

    lines = []

    max_str = format_decimal(max_val, 4)
    min_str = format_decimal(min_val, 4)
    max_label_len = max(len(max_str), len(min_str)) + 1

    for y in range(height):
        if y == 0:
            label = f"{max_str:>{max_label_len}}"
        elif y == height - 1:
            label = f"{min_str:>{max_label_len}}"
        else:
            label = " " * max_label_len

        row = "".join(grid[y])
        row_colored = ""
        for c in row:
            if c == "●":
                row_colored += f"[{COLORS['chart_train']}]{c}[/{COLORS['chart_train']}]"
            elif c == "◆":
                row_colored += f"[{COLORS['chart_eval']}]{c}[/{COLORS['chart_eval']}]"
            elif c == "│":
                row_colored += f"[dim]{c}[/dim]"
            else:
                row_colored += c

        lines.append(f"[dim]{label}[/dim] │ {row_colored}")

    x_axis = "─" * (width + 1)
    lines.append(f"{' ' * max_label_len} └{x_axis}")

    legend = f"[{COLORS['chart_train']}]●● Train[/{COLORS['chart_train']}]  [{COLORS['chart_eval']}]◆◆ Eval[/{COLORS['chart_eval']}]"
    lines.append(f"{' ' * (max_label_len + 2)}{legend}")

    min_loss = format_decimal(min(train_losses), 6)
    max_loss = format_decimal(max(train_losses), 6)
    lines.append(
        f"[dim]Cur: {format_decimal(current_val, 6)} | Min: {min_loss} | Max: {max_loss}[/dim]"
    )

    return "\n".join(lines)


def find_training_info() -> dict[str, Any]:
    info = {
        "running": False,
        "pid": None,
        "output_dir": None,
        "log_file": None,
        "status": "idle",
        "last_update": 0,
    }

    possible_dirs = [
        "./smollm_italian_improved",
        "./smollm_best_output",
        "./italian-gpt2-qlora-output",
    ]

    found_valid_process = False

    for dirname in possible_dirs:
        pid_file = os.path.join(dirname, ".training_pid")
        if os.path.exists(pid_file):
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                if os.path.exists(f"/proc/{pid}"):
                    log_file = os.path.join(dirname, "training.log")
                    if os.path.exists(log_file):
                        mtime = os.path.getmtime(log_file)
                        time_since_update = time.time() - mtime
                        info["last_update"] = mtime

                        if time_since_update < 300:
                            info["running"] = True
                            info["status"] = "running"
                            info["pid"] = pid
                            info["output_dir"] = dirname
                            info["log_file"] = log_file
                            found_valid_process = True
                        else:
                            info["running"] = False
                            info["status"] = "crashed"
                            info["pid"] = pid
                            info["output_dir"] = dirname
                            info["log_file"] = log_file
                            return info
            except (ValueError, FileNotFoundError, ProcessLookupError):
                info["running"] = False
                info["status"] = "crashed"
                info["output_dir"] = dirname
                info["log_file"] = os.path.join(dirname, "training.log")
                return info

    if found_valid_process:
        return info

    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split("\n"):
            if "train_italian" in line or "train_best" in line:
                if "python" in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        pid = int(parts[1])

                        for dirname in possible_dirs:
                            log_file = os.path.join(dirname, "training.log")
                            if os.path.exists(log_file):
                                mtime = os.path.getmtime(log_file)
                                time_since_update = time.time() - mtime
                                info["last_update"] = mtime

                                if time_since_update < 300:
                                    info["running"] = True
                                    info["status"] = "running"
                                    info["pid"] = pid
                                    info["output_dir"] = dirname
                                    info["log_file"] = log_file
                                    return info
                                else:
                                    info["running"] = True
                                    info["status"] = "running"
                                    info["pid"] = pid
                                    info["output_dir"] = dirname
                                    info["log_file"] = log_file
                                    return info
    except Exception:
        pass

    for dirname in possible_dirs:
        final_model = os.path.join(dirname, "pytorch_model.bin")
        if os.path.exists(final_model):
            info["status"] = "completed"
            info["output_dir"] = dirname
            info["log_file"] = os.path.join(dirname, "training.log")
            return info

    most_recent_dir = None
    most_recent_time = 0
    for dirname in possible_dirs:
        log_file = os.path.join(dirname, "training.log")
        if os.path.exists(log_file):
            mtime = os.path.getmtime(log_file)
            if mtime > most_recent_time:
                most_recent_time = mtime
                most_recent_dir = dirname
                info["last_update"] = mtime

    if most_recent_dir:
        info["output_dir"] = most_recent_dir
        info["log_file"] = os.path.join(most_recent_dir, "training.log")
        if time.time() - most_recent_time > 3600:
            info["status"] = "completed"
        else:
            info["status"] = "idle"

    return info


def parse_trainer_state(output_dir: str) -> dict[str, Any]:
    metrics = {
        "train_loss": [],
        "eval_loss": None,
        "learning_rate": None,
        "epoch": 0.0,
        "grad_norm": None,
        "total_steps": 5139,
        "current_step": 0,
    }

    try:
        checkpoint_dirs = []
        if os.path.exists(output_dir):
            for item in os.listdir(output_dir):
                if item.startswith("checkpoint-"):
                    checkpoint_path = os.path.join(output_dir, item)
                    if os.path.isdir(checkpoint_path):
                        try:
                            step = int(item.split("-")[1])
                            checkpoint_dirs.append((step, checkpoint_path))
                        except (ValueError, IndexError):
                            continue

        if not checkpoint_dirs:
            return metrics

        checkpoint_dirs.sort(key=lambda x: x[0])
        latest_checkpoint = checkpoint_dirs[-1][1]
        trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")

        if not os.path.exists(trainer_state_file):
            return metrics

        with open(trainer_state_file, "r") as f:
            trainer_state = json.load(f)

        log_history = trainer_state.get("log_history", [])

        metrics["total_steps"] = trainer_state.get("max_steps", 5139)
        metrics["current_step"] = trainer_state.get("global_step", 0)
        metrics["epoch"] = trainer_state.get("epoch", 0.0)

        train_losses = []
        last_train_entry = None

        for log_entry in log_history:
            if "loss" in log_entry and "eval_loss" not in log_entry:
                train_losses.append(log_entry["loss"])
                last_train_entry = log_entry
            if "eval_loss" in log_entry:
                metrics["eval_loss"] = log_entry["eval_loss"]

        metrics["train_loss"] = train_losses[-50:] if train_losses else []

        if last_train_entry:
            metrics["learning_rate"] = last_train_entry.get("learning_rate")
            metrics["grad_norm"] = last_train_entry.get("grad_norm")

    except Exception:
        pass

    return metrics


def parse_log_metrics(log_content: str) -> dict[str, Any]:
    metrics = {
        "train_loss": [],
        "eval_loss": None,
        "learning_rate": None,
        "epoch": 0.0,
        "grad_norm": None,
        "total_steps": 5139,
        "current_step": 0,
    }

    lines = log_content.split("\n")

    loss_pattern = r"['\"]loss['\"]\s*:\s*['\"]?([0-9.eE+-]+)['\"]?"
    eval_pattern = r"['\"]eval_loss['\"]\s*:\s*['\"]?([0-9.eE+-]+)['\"]?"
    lr_pattern = r"['\"]learning_rate['\"]\s*:\s*['\"]?([0-9.eE+-]+)['\"]?"
    epoch_pattern = r"['\"]epoch['\"]\s*:\s*['\"]?([0-9.eE+-]+)['\"]?"
    grad_pattern = r"['\"]grad_norm['\"]\s*:\s*['\"]?([0-9.eE+-]+)['\"]?"
    progress_pattern = r"\|\s*(\d+)/(\d+)\s+\["

    train_losses = []

    for line in lines:
        prog_match = re.search(progress_pattern, line)
        if prog_match:
            try:
                metrics["current_step"] = int(prog_match.group(1))
                metrics["total_steps"] = int(prog_match.group(2))
            except ValueError:
                pass

        if "loss" in line and "eval_loss" not in line:
            loss_match = re.search(loss_pattern, line)
            if loss_match:
                try:
                    loss_val = float(loss_match.group(1))
                    if loss_val < 100:
                        train_losses.append(loss_val)
                        metrics["train_loss"] = train_losses[-50:]
                except ValueError:
                    pass

        eval_match = re.search(eval_pattern, line)
        if eval_match:
            try:
                metrics["eval_loss"] = float(eval_match.group(1))
            except ValueError:
                pass

        lr_match = re.search(lr_pattern, line)
        if lr_match:
            try:
                metrics["learning_rate"] = float(lr_match.group(1))
            except ValueError:
                pass

        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            try:
                metrics["epoch"] = float(epoch_match.group(1))
            except ValueError:
                pass

        grad_match = re.search(grad_pattern, line)
        if grad_match:
            try:
                metrics["grad_norm"] = float(grad_match.group(1))
            except ValueError:
                pass

    return metrics


def get_gpu_metrics() -> dict[str, Any]:
    metrics = {
        "available": False,
        "name": "N/A",
        "vram_used": 0,
        "vram_total": 0,
        "utilization": 0,
        "temperature": 0,
    }

    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 5:
                metrics["available"] = True
                metrics["name"] = parts[0].strip()
                metrics["vram_used"] = float(parts[1].strip())
                metrics["vram_total"] = float(parts[2].strip())
                metrics["utilization"] = float(parts[3].strip())
                metrics["temperature"] = float(parts[4].strip())
    except Exception:
        pass

    return metrics


def get_system_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "cpu_percent": 0.0,
        "ram_used": 0.0,
        "ram_total": 0.0,
    }

    try:
        result = subprocess.run(
            ["top", "-bn1"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split("\n"):
            if "Cpu(s)" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if "id" in p:
                        try:
                            idle = float(parts[i - 1].replace(",", "."))
                            metrics["cpu_percent"] = round(100 - idle, 1)
                        except (ValueError, IndexError):
                            pass

        result = subprocess.run(
            ["free", "-m"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.split("\n"):
            if "Mem:" in line:
                parts = line.split()
                if len(parts) >= 3:
                    metrics["ram_used"] = float(parts[2])
                    metrics["ram_total"] = float(parts[1])
    except Exception:
        pass

    return metrics


def build_progress_bar(pct: float, width: int = 40) -> str:
    filled = int(pct / 100 * width)
    empty = width - filled

    progress_color = get_progress_color(pct)

    if pct < 30:
        gradient = COLORS["progress_low"]
    elif pct < 70:
        mid_point = (pct - 30) / 40
        gradient = COLORS["progress_mid"]
    else:
        gradient = COLORS["progress_high"]

    bar = f"[{gradient}]{'█' * filled}[/{gradient}][dim]{'░' * empty}[/dim]"
    return bar


def build_gpu_bar(util: float, width: int = 30) -> str:
    filled = int(util / 100 * width)
    empty = width - filled
    color = get_util_color(util)
    return f"[{color}]{'█' * filled}[/{color}][dim]{'░' * empty}[/dim]"


def format_eta(seconds: int) -> str:
    if seconds <= 0:
        return "N/A"

    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60

    if hours > 0:
        return f"~{hours}h {minutes}m"
    elif minutes > 0:
        return f"~{minutes}m {secs}s"
    return f"~{secs}s"


def main():
    os.system("cls" if os.name == "nt" else "clear")

    last_training_info = None
    refresh_interval = 3
    start_time = time.time()

    while True:
        os.system("cls" if os.name == "nt" else "clear")

        training_info = find_training_info()

        if training_info != last_training_info:
            last_training_info = training_info

        if not training_info["running"]:
            if training_info["status"] == "crashed":
                console.print(
                    Panel(
                        "[bold red]⚠️ TRAINING CRASHATO[/bold red]\n\n"
                        f"[yellow]Directory:[/yellow] {training_info['output_dir']}\n"
                        f"[yellow]PID:[/yellow] {training_info.get('pid', 'N/A')}\n\n"
                        "[cyan]Il training è terminato inaspettatamente.[/cyan]\n"
                        "Controlla il log per i dettagli dell'errore.\n\n"
                        "[dim]Riavviare con: python3 train_italian_improved.py --resume[/dim]",
                        title="📊 TRAINING MONITOR",
                        border_style="yellow",
                        box=box.DOUBLE,
                    )
                )
            elif training_info["status"] == "completed":
                console.print(
                    Panel(
                        "[bold green]✓ TRAINING COMPLETATO[/bold green]\n\n"
                        f"[yellow]Directory:[/yellow] {training_info['output_dir']}\n\n"
                        "[cyan]Il training è terminato con successo![/cyan]\n\n"
                        "[dim]Per testare il modello:[/dim]\n"
                        "  [cyan]python3 test_model.py[/cyan]",
                        title="📊 TRAINING MONITOR",
                        border_style="green",
                        box=box.DOUBLE,
                    )
                )
            else:
                console.print(
                    Panel(
                        "[bold red]Nessun training in corso[/bold red]\n\n"
                        "[yellow]Per avviare il training:[/yellow]\n"
                        "  [cyan]python3 train_italian_improved.py[/cyan]\n\n"
                        "[dim]Il monitor si aggiorna automaticamente...[/dim]",
                        title="📊 TRAINING MONITOR",
                        border_style="red",
                        box=box.DOUBLE,
                    )
                )
            time.sleep(refresh_interval)
            continue

        log_content = ""
        if training_info["log_file"]:
            try:
                with open(training_info["log_file"], "r") as f:
                    log_content = f.read()
            except Exception:
                pass

        log_metrics = parse_log_metrics(log_content)

        metrics = {
            "train_loss": [],
            "eval_loss": None,
            "learning_rate": None,
            "epoch": 0.0,
            "grad_norm": None,
            "total_steps": 5139,
            "current_step": 0,
        }

        if training_info.get("output_dir"):
            trainer_metrics = parse_trainer_state(training_info["output_dir"])
            if trainer_metrics.get("train_loss"):
                metrics["train_loss"] = trainer_metrics["train_loss"]
                metrics["eval_loss"] = trainer_metrics["eval_loss"]
                metrics["learning_rate"] = trainer_metrics["learning_rate"]
                metrics["grad_norm"] = trainer_metrics["grad_norm"]
                metrics["epoch"] = trainer_metrics["epoch"]
                metrics["total_steps"] = trainer_metrics["total_steps"]
                metrics["current_step"] = trainer_metrics["current_step"]

        if log_metrics.get("current_step", 0) > metrics.get("current_step", 0):
            metrics["current_step"] = log_metrics["current_step"]
            metrics["total_steps"] = log_metrics["total_steps"]

        if not metrics["train_loss"] and log_metrics.get("train_loss"):
            metrics["train_loss"] = log_metrics["train_loss"]
        if metrics["learning_rate"] is None and log_metrics.get("learning_rate"):
            metrics["learning_rate"] = log_metrics["learning_rate"]
        if metrics["grad_norm"] is None and log_metrics.get("grad_norm"):
            metrics["grad_norm"] = log_metrics["grad_norm"]
        if metrics["epoch"] == 0.0 and log_metrics.get("epoch"):
            metrics["epoch"] = log_metrics["epoch"]
        if metrics["eval_loss"] is None and log_metrics.get("eval_loss"):
            metrics["eval_loss"] = log_metrics["eval_loss"]

        if metrics["train_loss"]:
            loss_tracker.add_loss(metrics["train_loss"][-1])

        gpu = get_gpu_metrics()
        system = get_system_metrics()

        progress_pct = (
            (metrics["current_step"] / metrics["total_steps"] * 100)
            if metrics["total_steps"] > 0
            else 0
        )

        elapsed_time = int(time.time() - start_time)
        if progress_pct > 0 and elapsed_time > 10:
            total_estimated = int(elapsed_time / (progress_pct / 100))
            eta_seconds = total_estimated - elapsed_time
            eta_str = format_eta(eta_seconds)
        else:
            eta_str = "Calculating..."

        time_since_update = 0
        if training_info.get("last_update", 0) > 0:
            time_since_update = time.time() - training_info["last_update"]

        if time_since_update > 60:
            update_str = f"{int(time_since_update // 60)}m"
        else:
            update_str = f"{int(time_since_update)}s"

        trend_arrow, trend_color = loss_tracker.get_trend()
        change_rate = loss_tracker.get_change_rate()

        current_loss_str = (
            format_decimal(metrics["train_loss"][-1])
            if metrics["train_loss"]
            else "N/A"
        )
        eval_loss_str = (
            format_decimal(metrics["eval_loss"]) if metrics["eval_loss"] else "N/A"
        )
        lr_str = (
            format_decimal(metrics["learning_rate"])
            if metrics["learning_rate"]
            else "N/A"
        )
        grad_str = (
            format_decimal(metrics["grad_norm"]) if metrics["grad_norm"] else "N/A"
        )

        progress_bar = build_progress_bar(progress_pct, width=50)

        output = f"""[{COLORS["header"]}]╭────────────────── TRAINING MONITOR ──────────────────╮[/{COLORS["header"]}]
│ [{COLORS["success"]}]✅ Running[/{COLORS["success"]}] (PID: {training_info["pid"]}) │ Last update: [{COLORS["info"]}]{update_str}[/{COLORS["info"]}] ago      │
[{COLORS["header"]}]├──────────────────────────────────────────────────────┤[/{COLORS["header"]}]
│ [{COLORS["metric"]}]Progress[/{COLORS["metric"]}]                                            │
│ {progress_bar} [{get_progress_color(progress_pct)}]{progress_pct:.1f}%[/{get_progress_color(progress_pct)}]       │
│ Step [{COLORS["value"]}]{metrics["current_step"]}[/{COLORS["value"]}]/[{COLORS["value"]}]{metrics["total_steps"]}[/{COLORS["value"]}] │ Epoch [{COLORS["value"]}]{metrics["epoch"]:.2f}[/{COLORS["value"]}]/3 │ ETA: [{COLORS["info"]}]{eta_str}[/{COLORS["info"]}]           │
[{COLORS["header"]}]├──────────────────────────────────────────────────────┤[/{COLORS["header"]}]
│ [{COLORS["metric"]}]📊 Metrics[/{COLORS["metric"]}]                      [{COLORS["metric"]}]📈 Loss Trend[/{COLORS["metric"]}]          │"""

        chart = draw_loss_chart(
            metrics["train_loss"], metrics["eval_loss"], width=28, height=5
        )
        chart_lines = chart.split("\n")

        metric_lines = [
            f"Train Loss: [{trend_color}]{current_loss_str} {trend_arrow}[/{trend_color}]",
            f"Eval Loss:  [{COLORS['value']}]{eval_loss_str}[/{COLORS['value']}]",
            f"LR:         [{COLORS['value']}]{lr_str}[/{COLORS['value']}]",
            f"Grad Norm:  [{COLORS['value']}]{grad_str}[/{COLORS['value']}]",
        ]

        metrics_box_width = 28
        chart_box_width = 54 - metrics_box_width - 3

        output += f"\n│ ┌{'─' * (metrics_box_width - 2)}┐ {' ' * chart_box_width}│"

        for i, chart_line in enumerate(chart_lines):
            if i < len(metric_lines):
                metric_line = metric_lines[i]
                metric_content = f"│ {metric_line}"
                metric_padded = metric_content[:metrics_box_width].ljust(
                    metrics_box_width
                )
            else:
                metric_padded = "│" + " " * (metrics_box_width - 1)

            chart_stripped = chart_line.replace("[dim]", "").replace("[/dim]", "")
            chart_stripped = chart_stripped.replace(
                f"[{COLORS['chart_train']}]", ""
            ).replace(f"[/{COLORS['chart_train']}]", "")
            chart_stripped = chart_stripped.replace(
                f"[{COLORS['chart_eval']}]", ""
            ).replace(f"[/{COLORS['chart_eval']}]", "")
            chart_display = chart_line

            output += f"\n│ {metric_padded}│ {chart_display}"

        output += f"\n│ └{'─' * (metrics_box_width - 2)}┘ {' ' * chart_box_width}│"

        if gpu["available"]:
            vram_pct = (
                (gpu["vram_used"] / gpu["vram_total"] * 100)
                if gpu["vram_total"] > 0
                else 0
            )
            gpu_bar = build_gpu_bar(gpu["utilization"], width=20)
            temp_color = get_temp_color(gpu["temperature"])
            util_color = get_util_color(gpu["utilization"])
            vram_color = get_vram_color(vram_pct)

            output += f"""
[{COLORS["header"]}]├──────────────────────────────────────────────────────┤[/{COLORS["header"]}]
│ [{COLORS["metric"]}]🖥️ System Resources[/{COLORS["metric"]}]                               │
│ GPU: [{COLORS["value"]}]{markup.escape(gpu["name"])}[/{COLORS["value"]}]                                    │
│ VRAM: [{vram_color}]{gpu["vram_used"]:.0f}[/{vram_color}]/[{COLORS["value"]}]{gpu["vram_total"]:.0f}[/{COLORS["value"]}]MB ([{vram_color}]{vram_pct:.0f}%[/{vram_color}]) │ Util: [{util_color}]{gpu["utilization"]:.0f}%[/{util_color}]      │
│     {gpu_bar}             │
│ Temp: [{temp_color}]{gpu["temperature"]:.0f}°C[/{temp_color}] │ RAM: [{COLORS["value"]}]{system["ram_used"]:.0f}[/{COLORS["value"]}]/[{COLORS["value"]}]{system["ram_total"]:.0f}[/{COLORS["value"]}]MB │ CPU: [{COLORS["value"]}]{system["cpu_percent"]:.1f}%[/{COLORS["value"]}]          │"""
        else:
            ram_pct = (
                (system["ram_used"] / system["ram_total"] * 100)
                if system["ram_total"] > 0
                else 0
            )
            output += f"""
[{COLORS["header"]}]├──────────────────────────────────────────────────────┤[/{COLORS["header"]}]
│ [{COLORS["metric"]}]🖥️ System Resources[/{COLORS["metric"]}]                               │
│ GPU: [dim]Not available[/dim]                                     │
│ RAM: [{COLORS["value"]}]{system["ram_used"]:.0f}[/{COLORS["value"]}]/[{COLORS["value"]}]{system["ram_total"]:.0f}[/{COLORS["value"]}]MB ([{COLORS["value"]}]{ram_pct:.0f}%[/{COLORS["value"]}]) │ CPU: [{COLORS["value"]}]{system["cpu_percent"]:.1f}%[/{COLORS["value"]}]             │"""

        last_logs = []
        if training_info["log_file"] and os.path.exists(training_info["log_file"]):
            try:
                with open(training_info["log_file"], "r") as f:
                    lines = f.readlines()
                    last_logs = [l.strip() for l in lines[-2:] if l.strip()]
            except Exception:
                pass

        output += f"""
[{COLORS["header"]}]├──────────────────────────────────────────────────────┤[/{COLORS["header"]}]
│ [{COLORS["metric"]}]📝 Recent Log[/{COLORS["metric"]}]                                      │"""

        for line in last_logs[-2:]:
            truncated = line[:70] + "..." if len(line) > 70 else line
            output += f"\n│ [dim]{markup.escape(truncated)}[/dim]"

        output += f"""
[{COLORS["header"]}]╰──────────────────────────────────────────────────────╯[/{COLORS["header"]}]"""

        output += f"\n[dim]Ctrl+C to exit │ Refresh: {refresh_interval}s │ Dir: {markup.escape(training_info['output_dir'])}[/dim]"

        console.print(output)
        time.sleep(refresh_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Monitoraggio interrotto.[/bold yellow]")
