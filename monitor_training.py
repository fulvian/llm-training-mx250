#!/usr/bin/env python3
"""
Monitoraggio training in tempo reale con visualizzazione migliorata.
Include integrazione automatica di TensorBoard.

Usage:
    python3 monitor_training.py
"""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import urllib.error
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


# ============================================================================
# CONFIGURAZIONE
# ============================================================================

TENSORBOARD_PORT = 6006
TENSORBOARD_GRACE_PERIOD = 60  # secondi prima di stoppare dopo fine training
TENSORBOARD_RELOAD_INTERVAL = 30  # secondi
HEALTH_CHECK_INTERVAL = 30  # secondi tra health check

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
    "resume": "bold yellow",
    "tensorboard": "bold blue",
}

console = Console()


# ============================================================================
# TENSORBOARD MANAGER
# ============================================================================


def get_tailscale_ip() -> str:
    """Ottiene l'IP Tailscale del computer con fallback."""

    # Metodo 1: tailscale CLI (preferito)
    try:
        result = subprocess.run(
            ["tailscale", "ip"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split("\n")
            # Preferisci IPv4
            for ip in ips:
                if "." in ip and ip.startswith("100."):
                    return ip
            # Fallback al primo IP
            if ips:
                return ips[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Metodo 2: parsing interfaccia tailscale0
    try:
        result = subprocess.run(
            ["ip", "addr", "show", "tailscale0"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            import re

            match = re.search(r"inet\s+(\d+\.\d+\.\d+\.\d+)", result.stdout)
            if match:
                return match.group(1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Metodo 3: IP locale
    try:
        result = subprocess.run(
            ["hostname", "-I"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            if ips:
                return ips[0]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Fallback finale
    return "127.0.0.1"


def find_available_port(start_port: int = 6006) -> int:
    """Trova una porta disponibile a partire da start_port."""
    port = start_port
    max_attempts = 100

    for _ in range(max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("127.0.0.1", port))
                return port
        except OSError:
            port += 1

    return start_port  # Fallback


class TensorBoardManager:
    """Gestisce il ciclo di vita di TensorBoard."""

    def __init__(self, log_dir: str, port: int = TENSORBOARD_PORT):
        self.log_dir = log_dir
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self._last_health_check = 0
        self._is_healthy = False
        self._ip_address: Optional[str] = None
        self._training_stopped_time: Optional[float] = None

    def get_ip_address(self) -> str:
        """Ottiene l'IP per l'URL (cached)."""
        if self._ip_address is None:
            self._ip_address = get_tailscale_ip()
        return self._ip_address

    def get_url(self) -> str:
        """Ritorna l'URL completo di TensorBoard."""
        return f"http://{self.get_ip_address()}:{self.port}"

    def get_local_url(self) -> str:
        """Ritorna l'URL locale di TensorBoard."""
        return f"http://127.0.0.1:{self.port}"

    def is_running(self) -> bool:
        """Verifica se TensorBoard è in esecuzione."""
        if self.process is None:
            return False

        # Verifica che il processo esista
        if self.process.poll() is not None:
            self.process = None
            return False

        return True

    def _health_check(self) -> bool:
        """Verifica che TensorBoard risponda alle richieste HTTP."""
        try:
            url = f"http://127.0.0.1:{self.port}"
            response = urllib.request.urlopen(url, timeout=5)
            return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, Exception):
            return False

    def start(self) -> bool:
        """Avvia TensorBoard in background."""
        if self.is_running():
            return True

        # Trova porta disponibile
        self.port = find_available_port(self.port)

        # Verifica che la log_dir esista
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, exist_ok=True)

        cmd = [
            sys.executable,
            "-m",
            "tensorboard.main",
            "--logdir",
            self.log_dir,
            "--port",
            str(self.port),
            "--bind_all",
            "--reload_interval",
            str(TENSORBOARD_RELOAD_INTERVAL),
            "--reload_multifile=true",
        ]

        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                start_new_session=True,
            )

            # Attendi avvio
            time.sleep(2)

            # Verifica che sia partito
            if not self.is_running():
                return False

            # Health check
            self._is_healthy = self._health_check()
            self._last_health_check = time.time()

            return True

        except Exception as e:
            self.process = None
            return False

    def stop(self) -> None:
        """Ferma TensorBoard."""
        if self.process is not None:
            try:
                # Invia SIGTERM
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Se non termina, usa SIGKILL
                    self.process.kill()
                    self.process.wait(timeout=2)
            except Exception:
                pass
            finally:
                self.process = None
                self._is_healthy = False

    def update(self, training_running: bool) -> None:
        """
        Aggiorna lo stato di TensorBoard.
        Avvia se training attivo, ferma dopo grace period se no.
        """
        if training_running:
            self._training_stopped_time = None

            if not self.is_running():
                self.start()
            else:
                # Health check periodico
                now = time.time()
                if now - self._last_health_check > HEALTH_CHECK_INTERVAL:
                    self._is_healthy = self._health_check()
                    self._last_health_check = now

                    # Se non healthy, riavvia
                    if not self._is_healthy:
                        self.stop()
                        time.sleep(1)
                        self.start()
        else:
            # Training non attivo
            if self.is_running():
                if self._training_stopped_time is None:
                    self._training_stopped_time = time.time()
                elif (
                    time.time() - self._training_stopped_time > TENSORBOARD_GRACE_PERIOD
                ):
                    self.stop()

    def get_status(self) -> dict[str, Any]:
        """Ritorna lo stato completo di TensorBoard."""
        return {
            "running": self.is_running(),
            "healthy": self._is_healthy if self.is_running() else False,
            "url": self.get_url(),
            "local_url": self.get_local_url(),
            "port": self.port,
            "pid": self.process.pid if self.process else None,
            "ip": self.get_ip_address(),
        }

    def __del__(self):
        """Cleanup automatico."""
        self.stop()


# ============================================================================
# LOSS TRACKER
# ============================================================================


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


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


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


# ============================================================================
# TRAINING DETECTION
# ============================================================================


def find_training_info() -> dict[str, Any]:
    """Trova le informazioni sul training in corso.

    Strategia:
    1. Cerca un file .training_pid nelle directory di output
    2. Cerca processi Python in esecuzione con nomi di training
    3. Verifica che i log siano recenti (< 5 minuti)
    4. Restituisce la directory di output CORRETTA, non pattern glob
    """
    info = {
        "running": False,
        "pid": None,
        "output_dir": None,
        "log_file": None,
        "status": "idle",
        "last_update": 0,
        "log_dir": None,
        "training_start_time": None,
    }

    # Pattern per identificare i processi di training
    training_processes = [
        "train_qwen25_medical_italian",
        "train_qwen25_medical",
        "train_qwen25_qlora",
        "train_italian",
        "train_best",
        "train_qlora_optimized",
        "train_qlora_v2",
    ]

    # Directory di output possibili (pattern glob)
    output_patterns = [
        "./output_qwen25_medical_italian_*",
        "./output_qwen25_medical_*",
        "./output_qwen25_qlora_*",
        "./output_qwen_qlora_*",
        "./output_qlora_v2",
        "./output_qlora_optimized",
    ]

    # Log files possibili nella directory principale
    log_patterns = [
        "train_medical_italian.log",
        "train_medical_live.log",
        "train_medical_output.log",
        "train_output.log",
        "training_qwen25_*.log",
        "training_qwen*.log",
        "train_qlora_v2.log",
        "train_qlora_optimized.log",
    ]

    def expand_patterns(patterns: list) -> list:
        """Espande i pattern glob in directory reali."""
        import glob as glob_module

        expanded = set()
        for pattern in patterns:
            if "*" in pattern or "?" in pattern:
                matches = glob_module.glob(pattern)
                expanded.update(matches)
            elif os.path.exists(pattern):
                expanded.add(pattern)
        return list(expanded)

    def get_training_log_dir(output_dir: str) -> Optional[str]:
        """Trova la directory dei log TensorBoard per una data output_dir."""
        # Cerca direttamente nella output_dir
        logs_dir = os.path.join(output_dir, "logs")
        if os.path.exists(logs_dir):
            return logs_dir

        # Cerca nella parent directory con pattern
        parent = os.path.dirname(output_dir)
        if parent and parent != ".":
            for pattern in [
                f"{parent}/output_qwen25_medical_italian_*/logs",
                f"{parent}/output_qwen25_medical_*/logs",
                f"{parent}/output_qwen25_qlora_*/logs",
            ]:
                matches = expand_patterns([pattern])
                if matches:
                    # Prendi il più recente
                    return max(matches, key=os.path.getmtime)

        # Fallback a ./logs_* pattern
        logs_patterns = [
            "./logs_qwen25_medical_italian",
            "./logs_qwen25_medical",
            "./logs_qlora_v2",
            "./logs_qlora_optimized",
        ]
        for logs_dir in logs_patterns:
            if os.path.exists(logs_dir):
                return logs_dir

        return None

    # ========================================
    # FASE 1: Cerca .training_pid file
    # ========================================
    expanded_dirs = expand_patterns(output_patterns)

    for dirname in expanded_dirs:
        pid_file = os.path.join(dirname, ".training_pid")
        if os.path.exists(pid_file):
            try:
                with open(pid_file) as f:
                    pid = int(f.read().strip())
                if os.path.exists(f"/proc/{pid}"):
                    # Cerca il log file corretto
                    log_file = os.path.join(dirname, "training.log")
                    main_log = os.path.join(
                        os.path.dirname(dirname), "train_medical_italian.log"
                    )

                    actual_log = log_file if os.path.exists(log_file) else main_log

                    if os.path.exists(actual_log):
                        mtime = os.path.getmtime(actual_log)
                        pid_mtime = os.path.getmtime(pid_file)
                        time_since_update = time.time() - mtime
                        info["last_update"] = mtime
                        info["training_start_time"] = pid_mtime

                        if time_since_update < 300:
                            info["running"] = True
                            info["status"] = "running"
                            info["pid"] = pid
                            info["output_dir"] = dirname
                            info["log_file"] = actual_log
                            info["log_dir"] = get_training_log_dir(dirname)
                            return info
                        else:
                            info["running"] = False
                            info["status"] = "crashed"
                            info["pid"] = pid
                            info["output_dir"] = dirname
                            info["log_file"] = actual_log
                            info["log_dir"] = get_training_log_dir(dirname)
                            return info
            except (ValueError, FileNotFoundError, ProcessLookupError):
                pass

    # ========================================
    # FASE 2: Cerca processi in esecuzione
    # ========================================
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
        )

        active_pids = []
        for line in result.stdout.split("\n"):
            for proc_name in training_processes:
                if proc_name in line and "python" in line and "grep" not in line:
                    parts = line.split()
                    if len(parts) > 1:
                        try:
                            pid = int(parts[1])
                            active_pids.append((pid, proc_name))
                        except ValueError:
                            pass
                    break

        # Per ogni PID trovato, cerca la directory di output corretta
        for pid, proc_name in active_pids:
            # Metodo 1: Cerca nella directory di lavoro del processo
            try:
                cwd_link = f"/proc/{pid}/cwd"
                if os.path.exists(cwd_link):
                    proc_cwd = os.readlink(cwd_link)
                    # Il training potrebbe essere avviato dalla directory di output
                    for pattern in output_patterns:
                        matches = expand_patterns([pattern])
                        for match in matches:
                            if match in proc_cwd or proc_cwd in match:
                                # Verifica che ci sia un log file
                                log_file = os.path.join(match, "training.log")
                                main_log = "train_medical_italian.log"

                                actual_log = (
                                    log_file if os.path.exists(log_file) else main_log
                                )
                                if os.path.exists(actual_log):
                                    mtime = os.path.getmtime(actual_log)
                                    time_since_update = time.time() - mtime

                                    info["running"] = True
                                    info["status"] = (
                                        "running"
                                        if time_since_update < 300
                                        else "crashed"
                                    )
                                    info["pid"] = pid
                                    info["output_dir"] = match
                                    info["log_file"] = actual_log
                                    info["log_dir"] = get_training_log_dir(match)
                                    info["last_update"] = mtime
                                    return info
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass

            # Metodo 2: Cerca le directory di output più recenti e verifica con i log
            for dirname in sorted(expanded_dirs, key=os.path.getmtime, reverse=True):
                log_file = os.path.join(dirname, "training.log")
                main_log = os.path.join(
                    os.path.dirname(dirname), "train_medical_italian.log"
                )

                # Usa il log file più recente
                candidates = []
                if os.path.exists(log_file):
                    candidates.append((log_file, os.path.getmtime(log_file)))
                if os.path.exists(main_log):
                    candidates.append((main_log, os.path.getmtime(main_log)))

                if candidates:
                    actual_log, log_mtime = max(candidates, key=lambda x: x[1])
                    time_since_update = time.time() - log_mtime

                    # Se il log è stato aggiornato di recente (< 10 minuti)
                    if time_since_update < 600:
                        info["running"] = True
                        info["status"] = (
                            "running" if time_since_update < 300 else "crashed"
                        )
                        info["pid"] = pid
                        info["output_dir"] = dirname
                        info["log_file"] = actual_log
                        info["log_dir"] = get_training_log_dir(dirname)
                        info["last_update"] = log_mtime
                        return info
    except Exception:
        pass

    # ========================================
    # FASE 3: Cerca training completato o idle
    # ========================================

    # Cerca completamento (pytorch_model.bin)
    for dirname in expanded_dirs:
        final_model = os.path.join(dirname, "pytorch_model.bin")
        if os.path.exists(final_model):
            info["status"] = "completed"
            info["output_dir"] = dirname
            log_file = os.path.join(dirname, "training.log")
            main_log = os.path.join(
                os.path.dirname(dirname), "train_medical_italian.log"
            )
            info["log_file"] = log_file if os.path.exists(log_file) else main_log
            info["log_dir"] = get_training_log_dir(dirname)
            return info

    # Cerca la directory più recente con log file
    most_recent_dir = None
    most_recent_log_mtime = 0
    most_recent_log_file = None

    for dirname in expanded_dirs:
        log_file = os.path.join(dirname, "training.log")
        main_log = os.path.join(os.path.dirname(dirname), "train_medical_italian.log")

        for log in [log_file, main_log]:
            if os.path.exists(log):
                mtime = os.path.getmtime(log)
                if mtime > most_recent_log_mtime:
                    most_recent_log_mtime = mtime
                    most_recent_dir = dirname
                    most_recent_log_file = log

    if most_recent_dir:
        info["output_dir"] = most_recent_dir
        info["log_file"] = most_recent_log_file
        info["log_dir"] = get_training_log_dir(most_recent_dir)
        info["last_update"] = most_recent_log_mtime
        if time.time() - most_recent_log_mtime > 3600:
            info["status"] = "completed"
        else:
            info["status"] = "idle"

    return info


# ============================================================================
# METRICS PARSING
# ============================================================================


def parse_trainer_state(
    output_dir: str, ignore_old_checkpoints: bool = False
) -> dict[str, Any]:
    metrics = {
        "train_loss": [],
        "eval_loss": None,
        "learning_rate": None,
        "epoch": 0.0,
        "grad_norm": None,
        "total_steps": 0,  # Will be updated from trainer_state.json
        "current_step": 0,
        "starting_step": 0,
        "checkpoint_path": None,
    }

    if ignore_old_checkpoints:
        return metrics

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
        starting_step = checkpoint_dirs[-1][0]
        trainer_state_file = os.path.join(latest_checkpoint, "trainer_state.json")

        if not os.path.exists(trainer_state_file):
            return metrics

        with open(trainer_state_file, "r") as f:
            trainer_state = json.load(f)

        log_history = trainer_state.get("log_history", [])

        metrics["total_steps"] = trainer_state.get("max_steps", 5139)
        metrics["current_step"] = trainer_state.get("global_step", 0)
        metrics["epoch"] = trainer_state.get("epoch", 0.0)
        metrics["starting_step"] = starting_step
        metrics["checkpoint_path"] = latest_checkpoint

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
        "total_steps": 0,  # Will be updated from log parsing
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


# ============================================================================
# SYSTEM METRICS
# ============================================================================


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


# ============================================================================
# DISPLAY HELPERS
# ============================================================================


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


def build_tensorboard_section(tb_status: dict[str, Any], training_running: bool) -> str:
    """Costruisce la sezione TensorBoard per il display."""

    lines = []
    lines.append(
        f"[{COLORS['header']}]├──────────────────────────────────────────────────────┤[/{COLORS['header']}]"
    )
    lines.append(
        f"│ [{COLORS['tensorboard']}]📊 TensorBoard[/{COLORS['tensorboard']}]                                          │"
    )

    if tb_status["running"]:
        status_icon = "✅" if tb_status["healthy"] else "⚠️"
        status_color = COLORS["success"] if tb_status["healthy"] else COLORS["warning"]
        status_text = "Running" if tb_status["healthy"] else "Starting..."

        lines.append(
            f"│ Status: [{status_color}]{status_icon} {status_text}[/{status_color}]  │ Port: [{COLORS['value']}]{tb_status['port']}[/{COLORS['value']}]              │"
        )
        lines.append(
            f"│ [{COLORS['info']}]URL:[/{COLORS['info']}] [{COLORS['value']}]{tb_status['url']}[/{COLORS['value']}]                    │"
        )

        if tb_status.get("pid"):
            lines.append(
                f"│ PID: [{COLORS['value']}]{tb_status['pid']}[/{COLORS['value']}]                                          │"
            )
    else:
        lines.append(
            f"│ Status: [{COLORS['warning']}]⏸️ Not Running[/{COLORS['warning']}]                               │"
        )

        if training_running:
            lines.append(
                f"│ [{COLORS['info']}]Avvio automatico in corso...[/{COLORS['info']}]                        │"
            )
        else:
            lines.append(
                f"│ [{COLORS['info']}]Avvio manuale:[/{COLORS['info']}]                                       │"
            )
            lines.append(
                f"│   [dim]tensorboard --logdir=./logs_qlora_optimized[/dim]          │"
            )
            lines.append(
                f"│   [dim]--port 6006 --bind_all[/dim]                               │"
            )

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

# Variabile globale per cleanup
_tensorboard_manager: Optional[TensorBoardManager] = None


def cleanup_handler(signum, frame):
    """Handler per cleanup alla terminazione."""
    global _tensorboard_manager
    if _tensorboard_manager is not None:
        _tensorboard_manager.stop()
    console.print("\n[bold yellow]Monitoraggio interrotto.[/bold yellow]")
    sys.exit(0)


def main():
    global _tensorboard_manager

    os.system("cls" if os.name == "nt" else "clear")

    # Registra handler per cleanup
    signal.signal(signal.SIGINT, cleanup_handler)
    signal.signal(signal.SIGTERM, cleanup_handler)

    last_training_info = None
    refresh_interval = 3
    start_time = time.time()

    # Inizializza TensorBoard manager (cerca i log più recenti)
    import glob

    possible_tb_dirs = (
        glob.glob("./output_qwen25_medical_italian_*/logs")
        + glob.glob("./output_qwen25_medical_*/logs")
        + glob.glob("./output_qwen25_qlora_*/logs")
    )
    if possible_tb_dirs:
        tb_log_dir = max(possible_tb_dirs, key=os.path.getmtime)
    elif os.path.exists("./output_qwen_qlora_20260322_100639/logs"):
        tb_log_dir = "./output_qwen_qlora_20260322_100639/logs"
    elif os.path.exists("./output_qwen_qlora_20260322_100557/logs"):
        tb_log_dir = "./output_qwen_qlora_20260322_100557/logs"
    elif os.path.exists("./logs_qlora_v2"):
        tb_log_dir = "./logs_qlora_v2"
    else:
        tb_log_dir = "./logs_qlora_optimized"
    tb_manager = TensorBoardManager(tb_log_dir, TENSORBOARD_PORT)
    _tensorboard_manager = tb_manager

    while True:
        os.system("cls" if os.name == "nt" else "clear")

        training_info = find_training_info()

        # Aggiorna TensorBoard
        tb_manager.update(training_info["running"])
        tb_status = tb_manager.get_status()

        if training_info != last_training_info:
            last_training_info = training_info

        # ====================================================================
        # DISPLAY: NO TRAINING
        # ====================================================================
        if not training_info["running"]:
            # Costruisci sezione TensorBoard
            tb_section = build_tensorboard_section(tb_status, training_info["running"])

            if training_info["status"] == "crashed":
                console.print(
                    Panel(
                        "[bold red]⚠️ TRAINING CRASHATO[/bold red]\n\n"
                        f"[yellow]Directory:[/yellow] {training_info['output_dir']}\n"
                        f"[yellow]PID:[/yellow] {training_info.get('pid', 'N/A')}\n\n"
                        "[cyan]Il training è terminato inaspettatamente.[/cyan]\n"
                        "Controlla il log per i dettagli dell'errore.\n\n"
                        "[dim]Riavviare con: python3 train_italian_improved.py --resume[/dim]\n\n"
                        f"{tb_section}",
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
                        "  [cyan]python3 test_model.py[/cyan]\n\n"
                        f"{tb_section}",
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
                        "[dim]Il monitor si aggiorna automaticamente...[/dim]\n\n"
                        f"{tb_section}",
                        title="📊 TRAINING MONITOR",
                        border_style="red",
                        box=box.DOUBLE,
                    )
                )
            time.sleep(refresh_interval)
            continue

        # ====================================================================
        # DISPLAY: TRAINING RUNNING
        # ====================================================================
        log_content = ""
        if training_info["log_file"]:
            try:
                with open(training_info["log_file"], "r") as f:
                    log_content = f.read()
            except Exception:
                pass

        log_metrics = parse_log_metrics(log_content)

        training_is_fresh = False
        if training_info.get("training_start_time"):
            time_since_start = time.time() - training_info["training_start_time"]
            training_is_fresh = time_since_start < 300

        metrics = {
            "train_loss": [],
            "eval_loss": None,
            "learning_rate": None,
            "epoch": 0.0,
            "grad_norm": None,
            "total_steps": 0,  # Will be updated from trainer_state.json or log
            "current_step": 0,
            "starting_step": 0,
            "checkpoint_path": None,
            "data_source": "checkpoint",
            "is_new_training": training_is_fresh,
        }

        if training_info.get("output_dir"):
            trainer_metrics = parse_trainer_state(
                training_info["output_dir"],
                ignore_old_checkpoints=training_is_fresh,
            )
            metrics["starting_step"] = trainer_metrics.get("starting_step", 0)
            metrics["checkpoint_path"] = trainer_metrics.get("checkpoint_path")
            if trainer_metrics.get("train_loss"):
                metrics["train_loss"] = trainer_metrics["train_loss"]
                metrics["eval_loss"] = trainer_metrics["eval_loss"]
                metrics["learning_rate"] = trainer_metrics["learning_rate"]
                metrics["grad_norm"] = trainer_metrics["grad_norm"]
                metrics["epoch"] = trainer_metrics["epoch"]
                metrics["total_steps"] = trainer_metrics["total_steps"]
                metrics["current_step"] = trainer_metrics["current_step"]

        if training_is_fresh:
            metrics["data_source"] = "live (new training)"

        if log_metrics.get("current_step", 0) > 0:
            metrics["data_source"] = "live"
            metrics["current_step"] = log_metrics["current_step"]
            metrics["total_steps"] = log_metrics["total_steps"]

        if log_metrics.get("train_loss"):
            metrics["data_source"] = "live"
            metrics["train_loss"] = log_metrics["train_loss"]
        if log_metrics.get("learning_rate"):
            metrics["data_source"] = "live"
            metrics["learning_rate"] = log_metrics["learning_rate"]
        if log_metrics.get("grad_norm"):
            metrics["data_source"] = "live"
            metrics["grad_norm"] = log_metrics["grad_norm"]
        if log_metrics.get("epoch"):
            metrics["data_source"] = "live"
            metrics["epoch"] = log_metrics["epoch"]
        if log_metrics.get("eval_loss"):
            metrics["data_source"] = "live"
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

        resume_indicator = ""
        if metrics["starting_step"] > 0 and metrics["data_source"] == "checkpoint":
            resume_indicator = f" [{COLORS['resume']}]🔄 RESUMED from step {metrics['starting_step']}[/{COLORS['resume']}]"

        new_training_indicator = ""
        if metrics.get("is_new_training"):
            new_training_indicator = (
                f" [{COLORS['success']}][NEW][/{COLORS['success']}]"
            )

        # ====================================================================
        # BUILD OUTPUT
        # ====================================================================
        output = f"""[{COLORS["header"]}]╭────────────────── TRAINING MONITOR ──────────────────╮[/{COLORS["header"]}]
│ [{COLORS["success"]}]✅ Running[/{COLORS["success"]}] (PID: {training_info["pid"]}){resume_indicator}{new_training_indicator} │ Last update: [{COLORS["info"]}]{update_str}[/{COLORS["info"]}] ago   │
[{COLORS["header"]}]├──────────────────────────────────────────────────────┤[/{COLORS["header"]}]
│ [{COLORS["metric"]}]Progress[/{COLORS["metric"]}]                                            │
│ {progress_bar} [{get_progress_color(progress_pct)}]{progress_pct:.1f}%[/{get_progress_color(progress_pct)}]       │
│ Step [{COLORS["value"]}]{metrics["current_step"]}[/{COLORS["value"]}]/[{COLORS["value"]}]{metrics["total_steps"]}[/{COLORS["value"]}] (src: {metrics["data_source"]}) │ Epoch [{COLORS["value"]}]{metrics["epoch"]:.2f}[/{COLORS["value"]}]/3 │ ETA: [{COLORS["info"]}]{eta_str}[/{COLORS["info"]}]        │
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

        # System Resources
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
│ RAM: [{COLORS["value"]}]{system["ram_used"]:.0f}[/{COLORS["value"]}]/[{COLORS["value"]}]{system["ram_total"]:.0f}[/{COLORS["value"]}]MB ([{COLORS["value"]}]{ram_pct:.0f}%[/{COLORS["value"]}) │ CPU: [{COLORS["value"]}]{system["cpu_percent"]:.1f}%[/{COLORS["value"]}]             │"""

        # TensorBoard Section
        output += "\n" + build_tensorboard_section(tb_status, training_info["running"])

        # Recent Log
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
    main()
