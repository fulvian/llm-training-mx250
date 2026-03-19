#!/usr/bin/env python3
"""
Monitoraggio training in tempo reale.
Trova automaticamente il training in corso e mostra metriche in tempo reale.

Usage:
    python3 monitor_training.py
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import os

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
except ImportError:
    print("Installare rich: pip install rich")
    sys.exit(1)


console = Console()


def find_training_info() -> dict[str, Any]:
    """Trova automaticamente le informazioni sul training in corso."""
    info = {
        "running": False,
        "pid": None,
        "output_dir": None,
        "log_file": None,
    }

    possible_dirs = [
        "./smollm_italian_improved",
        "./smollm_best_output",
        "./italian-gpt2-qlora-output",
    ]

    # First, try to find by PID file
    for dirname in possible_dirs:
        pid_file = os.path.join(dirname, ".training_pid")
        if os.path.exists(pid_file):
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                if os.path.exists(f"/proc/{pid}"):
                    info["running"] = True
                    info["pid"] = pid
                    info["output_dir"] = dirname
                    info["log_file"] = os.path.join(dirname, "training.log")
                    return info
            except (ValueError, FileNotFoundError):
                continue

    # Try to find by process name
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

                        # Find the correct output dir by checking for training.log
                        for dirname in possible_dirs:
                            log_file = os.path.join(dirname, "training.log")
                            if os.path.exists(log_file):
                                # Check if this log file is being written to recently
                                mtime = os.path.getmtime(log_file)
                                if (
                                    time.time() - mtime < 300
                                ):  # Modified in last 5 minutes
                                    info["running"] = True
                                    info["pid"] = pid
                                    info["output_dir"] = dirname
                                    info["log_file"] = log_file
                                    return info

                        # If no recent log found, use the first existing dir
                        for dirname in possible_dirs:
                            if os.path.exists(dirname):
                                info["running"] = True
                                info["pid"] = pid
                                info["output_dir"] = dirname
                                info["log_file"] = os.path.join(dirname, "training.log")
                                return info
    except Exception:
        pass

    # Fallback: check for most recently modified training.log
    most_recent_dir = None
    most_recent_time = 0
    for dirname in possible_dirs:
        log_file = os.path.join(dirname, "training.log")
        if os.path.exists(log_file):
            mtime = os.path.getmtime(log_file)
            if mtime > most_recent_time:
                most_recent_time = mtime
                most_recent_dir = dirname

    if most_recent_dir:
        info["output_dir"] = most_recent_dir
        info["log_file"] = os.path.join(most_recent_dir, "training.log")

    return info


def parse_log_metrics(log_content: str) -> dict[str, Any]:
    """Estrae le metriche dal contenuto del log."""
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

    # Regex patterns - support multiple formats
    # Format: 'loss': 1.234 or "loss": 1.234 or loss: 1.234
    # Also: Metric: loss = 1.234
    loss_pattern = r"(?:Metric:\s*)?['\"]?loss['\"]?:\s*([0-9.]+)"
    eval_pattern = r"(?:Metric:\s*)?['\"]?eval_loss['\"]?:\s*([0-9.]+)"
    lr_pattern = r"(?:Metric:\s*)?['\"]?learning_rate['\"]?:\s*([0-9.e-]+)"
    epoch_pattern = r"(?:Metric:\s*)?['\"]?epoch['\"]?:\s*([0-9.e-]+)"
    grad_pattern = r"(?:Metric:\s*)?['\"]?grad_norm['\"]?:\s*([0-9.]+)"
    # Progress: | 10/5139 [06:53<58:55:48, 41.36s/it]
    progress_pattern = r"\|\s*(\d+)/(\d+)\s+\["

    train_losses = []

    for line in lines:
        # Progress bar - this is the main way to get current step
        prog_match = re.search(progress_pattern, line)
        if prog_match:
            try:
                metrics["current_step"] = int(prog_match.group(1))
                metrics["total_steps"] = int(prog_match.group(2))
            except ValueError:
                pass

        # Try to find loss values in log lines (not progress bar)
        # Format: {"loss": 1.234, "learning_rate": 1e-5, ...}
        if "loss" in line and "eval_loss" not in line:
            loss_match = re.search(loss_pattern, line)
            if loss_match:
                try:
                    loss_val = float(loss_match.group(1))
                    if loss_val < 100:  # Sanity check - loss shouldn't be > 100
                        train_losses.append(loss_val)
                        metrics["train_loss"] = train_losses[-50:]
                except ValueError:
                    pass

        # Eval loss
        eval_match = re.search(eval_pattern, line)
        if eval_match:
            try:
                metrics["eval_loss"] = float(eval_match.group(1))
            except ValueError:
                pass

        # Learning rate
        lr_match = re.search(lr_pattern, line)
        if lr_match:
            try:
                metrics["learning_rate"] = float(lr_match.group(1))
            except ValueError:
                pass

        # Epoch
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            try:
                metrics["epoch"] = float(epoch_match.group(1))
            except ValueError:
                pass

        # Gradient norm
        grad_match = re.search(grad_pattern, line)
        if grad_match:
            try:
                metrics["grad_norm"] = float(grad_match.group(1))
            except ValueError:
                pass

    return metrics


def get_gpu_metrics() -> dict[str, Any]:
    """Ottiene metriche GPU."""
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
    """Ottiene metriche di sistema."""
    metrics = {
        "cpu_percent": 0,
        "ram_used": 0,
        "ram_total": 0,
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


def draw_loss_chart(
    train_losses: list[float],
    eval_loss: Optional[float],
    width: int = 35,
    height: int = 6,
) -> str:
    """Disegna un grafico ASCII del trend della loss."""
    if not train_losses:
        return "Nessun dato disponibile"

    all_losses = train_losses.copy()
    if eval_loss is not None:
        all_losses.append(eval_loss)

    if not all_losses or max(all_losses) == 0:
        return "Dati insufficienti"

    min_val = min(all_losses)
    max_val = max(all_losses)
    val_range = max_val - min_val if max_val != min_val else 1

    def get_y_pos(val: float, h: int) -> int:
        return h - 1 - int(((val - min_val) / val_range) * (h - 1))

    # Crea griglia
    grid = [[" " for _ in range(len(train_losses))] for _ in range(height + 1)]

    # Plotta punti train
    for i, loss in enumerate(train_losses):
        y = get_y_pos(loss, height)
        if 0 <= y <= height:
            grid[y][i] = "▓"

    # Plotta eval se presente
    if eval_loss is not None:
        eval_y = get_y_pos(eval_loss, height)
        if 0 <= eval_y <= height and len(train_losses) > 0:
            grid[eval_y][len(train_losses) - 1] = "◆"

    # Disegna bordi
    result = ""
    for y in range(height, -1, -1):
        line = ""
        for x, cell in enumerate(grid[y]):
            if y == 0 or y == height:
                line += "─" if cell == " " else cell
            else:
                line += "│" if x == 0 else cell
        result += line + "\n"

    return result.strip()


def main():
    """Main loop del monitor."""
    # Clear iniziale
    os.system("cls" if os.name == "nt" else "clear")

    last_training_info = None
    refresh_interval = 3

    while True:
        # Clear screen ad ogni ciclo per evitare duplicati
        os.system("cls" if os.name == "nt" else "clear")

        training_info = find_training_info()

        if training_info != last_training_info:
            last_training_info = training_info

        if not training_info["running"]:
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

        metrics = parse_log_metrics(log_content)
        gpu = get_gpu_metrics()
        system = get_system_metrics()

        progress_pct = (
            (metrics["current_step"] / metrics["total_steps"] * 100)
            if metrics["total_steps"] > 0
            else 0
        )

        status_color = "green"
        status_text = f"● Attivo (PID: {training_info['pid']})"

        current_loss = (
            f"{metrics['train_loss'][-1]:.6f}" if metrics["train_loss"] else "N/A"
        )
        eval_loss_str = f"{metrics['eval_loss']:.6f}" if metrics["eval_loss"] else "N/A"
        lr_str = (
            f"{metrics['learning_rate']:.2e}" if metrics["learning_rate"] else "N/A"
        )
        grad_str = f"{metrics['grad_norm']:.6f}" if metrics["grad_norm"] else "N/A"

        vram_str = "N/A"
        gpu_util_str = "N/A"
        temp_str = "N/A"
        if gpu["available"]:
            vram_pct = (
                (gpu["vram_used"] / gpu["vram_total"] * 100)
                if gpu["vram_total"] > 0
                else 0
            )
            vram_str = (
                f"{gpu['vram_used']:.0f}/{gpu['vram_total']:.0f}MB ({vram_pct:.0f}%)"
            )
            gpu_util_str = f"{gpu['utilization']:.0f}%"
            temp_str = f"{gpu['temperature']:.0f}°C"

        ram_pct = (
            (system["ram_used"] / system["ram_total"] * 100)
            if system["ram_total"] > 0
            else 0
        )
        ram_str = (
            f"{system['ram_used']:.0f}/{system['ram_total']:.0f}MB ({ram_pct:.0f}%)"
        )
        cpu_str = f"{system['cpu_percent']:.1f}%"

        chart = draw_loss_chart(metrics["train_loss"], metrics["eval_loss"])

        last_logs = []
        if training_info["log_file"] and os.path.exists(training_info["log_file"]):
            try:
                with open(training_info["log_file"], "r") as f:
                    lines = f.readlines()
                    last_logs = [l.strip() for l in lines[-3:] if l.strip()]
            except Exception:
                pass

        progress_bar = "█" * int(progress_pct // 2) + "░" * (
            50 - int(progress_pct // 2)
        )

        output = f"""[bold cyan]TRAINING MONITOR[/bold cyan]                    [bold {status_color}]{status_text}[/bold {status_color}]

Progresso: Step {metrics["current_step"]}/{metrics["total_steps"]} ({progress_pct:.1f}%)  |  Epoch {metrics["epoch"]:.2f}/3
{progress_bar}

[bold]Trend Loss[/bold]                      [bold]Metriche[/bold]
{chart:<40} Train: {current_loss}
                               Eval:  {eval_loss_str}
                               LR:    {lr_str}
                               Grad:  {grad_str}

[bold]Risorse:[/bold]
  GPU VRAM: {vram_str:<25} GPU Util: {gpu_util_str}
  GPU Temp: {temp_str:<25} CPU: {cpu_str}
  RAM: {ram_str}

[bold]Log Recenti:[/bold]
"""
        for line in last_logs[-3:]:
            output += f"  {line[:80]}\n"

        output += f"\nCtrl+C per uscire | Aggiornamento: {refresh_interval}s"

        panel = Panel(
            output,
            border_style="blue",
            title=f"Training: {training_info['output_dir']}",
            padding=(0, 1),
        )

        console.print(panel)
        time.sleep(refresh_interval)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Monitoraggio interrotto.[/bold yellow]")
