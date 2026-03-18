#!/usr/bin/env python3
"""
GPU and Training Monitor for LLM Training
Monitors GPU metrics and training progress in real-time.
Supports live monitoring of active training processes.
"""

import argparse
import json
import os
import re
import struct
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

try:
    import pynvml

    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False

try:
    import psutil

    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import subprocess

    NVIDIA_SMI_AVAILABLE = True
except ImportError:
    NVIDIA_SMI_AVAILABLE = False


from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


@dataclass
class GPUInfo:
    index: int = 0
    name: str = "N/A"
    utilization: float = 0.0
    memory_used: float = 0.0
    memory_total: float = 0.0
    temperature: int = 0
    power_usage: float = 0.0
    power_limit: float = 0.0
    clock_core: int = 0
    clock_memory: int = 0
    fan_speed: int = 0
    processes: list = field(default_factory=list)

    @property
    def memory_percent(self) -> float:
        if self.memory_total > 0:
            return (self.memory_used / self.memory_total) * 100
        return 0.0

    @property
    def power_percent(self) -> float:
        if self.power_limit > 0:
            return (self.power_usage / self.power_limit) * 100
        return 0.0


@dataclass
class TrainingMetrics:
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    step: Optional[int] = None
    grad_norm: Optional[float] = None
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    samples_per_second: Optional[float] = None
    loss_history: list = field(default_factory=list)
    last_update: Optional[datetime] = None


@dataclass
class ProcessInfo:
    pid: int = 0
    command: str = ""
    cmdline: list = field(default_factory=list)
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    memory_percent: float = 0.0
    read_bytes: int = 0
    write_bytes: int = 0
    read_mb: float = 0.0
    write_mb: float = 0.0
    create_time: Optional[datetime] = None
    status: str = ""
    is_training: bool = False


@dataclass
class MonitorStatus:
    is_live: bool = False
    source_type: str = "none"
    source_path: Optional[str] = None
    process_pid: Optional[int] = None
    process_command: Optional[str] = None
    log_file: Optional[str] = None
    tensorboard_dir: Optional[str] = None


class NVSMIFallback:
    """Fallback GPU monitor using nvidia-smi when pynvml is not available."""

    def __init__(self):
        self.initialized = False
        self.gpu_count = 0

        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                self.gpu_count = len(result.stdout.strip().split("\n"))
                self.initialized = True
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    def get_gpu_info(self, index: int) -> GPUInfo:
        if not self.initialized or index >= self.gpu_count:
            return GPUInfo(index=index)

        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit,clocks.gr,clocks.mem,fan.speed",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )

            if result.returncode != 0:
                return GPUInfo(index=index)

            lines = result.stdout.strip().split("\n")
            if index >= len(lines):
                return GPUInfo(index=index)

            parts = [p.strip() for p in lines[index].split(",")]

            if len(parts) < 5:
                return GPUInfo(index=index)

            info = GPUInfo(index=index)
            info.name = parts[1] if len(parts) > 1 else "Unknown"
            info.utilization = (
                float(parts[2]) if parts[2] not in ("N/A", "[N/A]") else 0.0
            )
            info.memory_used = (
                float(parts[3]) / 1024 if parts[3] not in ("N/A", "[N/A]") else 0.0
            )
            info.memory_total = (
                float(parts[4]) / 1024 if parts[4] not in ("N/A", "[N/A]") else 0.0
            )
            info.temperature = int(parts[5]) if parts[5] not in ("N/A", "[N/A]") else 0

            if len(parts) > 6 and parts[6] not in ("N/A", "[N/A]"):
                info.power_usage = (
                    float(parts[6].replace("[", "").replace("]", ""))
                    if parts[6].replace("[", "").replace("]", "") != "N/A"
                    else 0.0
                )
            if len(parts) > 7 and parts[7] not in ("N/A", "[N/A]"):
                info.power_limit = float(parts[7])

            if len(parts) > 8 and parts[8] not in ("N/A", "[N/A]"):
                info.clock_core = int(parts[8])
            if len(parts) > 9 and parts[9] not in ("N/A", "[N/A]"):
                info.clock_memory = int(parts[9])

            if len(parts) > 10 and parts[10] not in ("N/A", "[N/A]"):
                info.fan_speed = int(parts[10])

            try:
                proc_result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if proc_result.returncode == 0:
                    info.processes = []
                    for line in proc_result.stdout.strip().split("\n"):
                        if line.strip():
                            proc_parts = [p.strip() for p in line.split(",")]
                            if len(proc_parts) >= 2:
                                info.processes.append(
                                    {
                                        "pid": int(proc_parts[0]),
                                        "memory": float(proc_parts[1]) / 1024
                                        if proc_parts[1] not in ("N/A", "[N/A]")
                                        else 0.0,
                                    }
                                )
            except subprocess.SubprocessError:
                pass

            return info

        except (subprocess.SubprocessError, ValueError, IndexError):
            return GPUInfo(index=index)


class GPUMonitor:
    def __init__(self):
        self.initialized = False
        self.gpu_count = 0
        self.gpu_handles = []
        self.smi_fallback = None

        if NVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                self.initialized = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
                self.gpu_handles = [
                    pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(self.gpu_count)
                ]
            except pynvml.NVMLError:
                self.initialized = False

        if not self.initialized and NVIDIA_SMI_AVAILABLE:
            self.smi_fallback = NVSMIFallback()
            if self.smi_fallback.initialized:
                self.gpu_count = self.smi_fallback.gpu_count

    def get_gpu_info(self, index: int) -> GPUInfo:
        if index >= self.gpu_count:
            return GPUInfo(index=index)

        if self.smi_fallback is not None and self.smi_fallback.initialized:
            return self.smi_fallback.get_gpu_info(index)

        if not self.initialized:
            return GPUInfo(index=index)

        try:
            handle = self.gpu_handles[index]
            info = GPUInfo(index=index)

            info.name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(info.name, bytes):
                info.name = info.name.decode("utf-8")

            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                info.utilization = util.gpu
            except pynvml.NVMLError:
                pass

            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                info.memory_used = mem.used / (1024**3)
                info.memory_total = mem.total / (1024**3)
            except pynvml.NVMLError:
                pass

            try:
                info.temperature = pynvml.nvmlDeviceGetTemperature(
                    handle, pynvml.NVML_TEMPERATURE_GPU
                )
            except pynvml.NVMLError:
                pass

            try:
                info.power_usage = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                info.power_limit = (
                    pynvml.nvmlDeviceGetPowerManagementLimit(handle) / 1000.0
                )
            except pynvml.NVMLError:
                pass

            try:
                info.clock_core = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_GRAPHICS
                )
                info.clock_memory = pynvml.nvmlDeviceGetClockInfo(
                    handle, pynvml.NVML_CLOCK_MEM
                )
            except pynvml.NVMLError:
                pass

            try:
                info.fan_speed = pynvml.nvmlDeviceGetFanSpeed(handle)
            except pynvml.NVMLError:
                pass

            try:
                procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                info.processes = [
                    {
                        "pid": p.pid,
                        "memory": p.usedGpuMemory / (1024**3) if p.usedGpuMemory else 0,
                    }
                    for p in procs
                ]
            except pynvml.NVMLError:
                pass

            return info

        except pynvml.NVMLError:
            return GPUInfo(index=index)

    def get_all_gpu_info(self) -> list[GPUInfo]:
        return [self.get_gpu_info(i) for i in range(self.gpu_count)]

    def shutdown(self):
        if self.initialized and NVML_AVAILABLE:
            try:
                pynvml.nvmlShutdown()
            except pynvml.NVMLError:
                pass


class ProcessMonitor:
    """Monitor active training processes."""

    TRAINING_KEYWORDS = [
        "train",
        "torchrun",
        "accelerate",
        "deepspeed",
        "transformers",
        "finetune",
        "lora",
        "qlora",
        "sft",
        "rlhf",
        "trl",
        "axolotl",
        "llama",
        "mistral",
        "gpt",
        "bert",
        "whisper",
        "diffusers",
    ]

    def __init__(self):
        self.training_pid: Optional[int] = None
        self.process: Optional[psutil.Process] = None
        self.last_io_counters = None
        self.last_io_time = None

    def find_training_processes(self) -> list[ProcessInfo]:
        if not PSUTIL_AVAILABLE:
            return []

        processes = []
        gpu_pids = self._get_gpu_process_pids()

        for proc in psutil.process_iter(
            ["pid", "name", "cmdline", "create_time", "status"]
        ):
            try:
                info = ProcessInfo()
                info.pid = proc.info["pid"]
                info.cmdline = proc.info.get("cmdline") or []
                info.command = (
                    " ".join(info.cmdline) if info.cmdline else proc.info["name"]
                )
                info.create_time = datetime.fromtimestamp(proc.info["create_time"])
                info.status = proc.info.get("status", "")

                cmdline_str = " ".join(info.cmdline).lower() if info.cmdline else ""
                is_training = any(kw in cmdline_str for kw in self.TRAINING_KEYWORDS)
                is_gpu_process = info.pid in gpu_pids
                is_python = "python" in proc.info["name"].lower()

                info.is_training = is_training or (is_python and is_gpu_process)

                if info.is_training:
                    try:
                        p = psutil.Process(info.pid)
                        info.cpu_percent = p.cpu_percent(interval=0.1)
                        mem_info = p.memory_info()
                        info.memory_mb = mem_info.rss / (1024 * 1024)
                        info.memory_percent = p.memory_percent()

                        try:
                            io_info = p.io_counters()
                            info.read_bytes = io_info.read_bytes
                            info.write_bytes = io_info.write_bytes
                            info.read_mb = io_info.read_bytes / (1024 * 1024)
                            info.write_mb = io_info.write_bytes / (1024 * 1024)
                        except (psutil.AccessDenied, AttributeError):
                            pass
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue

                    processes.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue

        processes.sort(key=lambda p: p.memory_mb, reverse=True)
        return processes

    def _get_gpu_process_pids(self) -> set:
        pids = set()

        if NVML_AVAILABLE:
            try:
                for i in range(pynvml.nvmlDeviceGetCount()):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    procs = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for p in procs:
                        pids.add(p.pid)
            except pynvml.NVMLError:
                pass

        if not pids and NVIDIA_SMI_AVAILABLE:
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-compute-apps=pid",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    for line in result.stdout.strip().split("\n"):
                        if line.strip().isdigit():
                            pids.add(int(line.strip()))
            except subprocess.SubprocessError:
                pass

        return pids

    def attach_to_process(self, pid: int) -> bool:
        if not PSUTIL_AVAILABLE:
            return False

        try:
            self.process = psutil.Process(pid)
            self.training_pid = pid
            self.last_io_counters = (
                self.process.io_counters()
                if hasattr(self.process, "io_counters")
                else None
            )
            self.last_io_time = time.time()
            return True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.process = None
            self.training_pid = None
            return False

    def get_process_info(self) -> Optional[ProcessInfo]:
        if not self.process or not PSUTIL_AVAILABLE:
            return None

        try:
            info = ProcessInfo()
            info.pid = self.training_pid
            info.cmdline = self.process.cmdline()
            info.command = " ".join(info.cmdline) if info.cmdline else ""
            info.status = self.process.status()
            info.is_training = True

            info.cpu_percent = self.process.cpu_percent(interval=0.1)
            mem_info = self.process.memory_info()
            info.memory_mb = mem_info.rss / (1024 * 1024)
            info.memory_percent = self.process.memory_percent()

            try:
                io_info = self.process.io_counters()
                info.read_bytes = io_info.read_bytes
                info.write_bytes = io_info.write_bytes
                info.read_mb = io_info.read_bytes / (1024 * 1024)
                info.write_mb = io_info.write_bytes / (1024 * 1024)
            except (psutil.AccessDenied, AttributeError):
                pass

            return info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self.process = None
            self.training_pid = None
            return None

    def get_open_log_files(self) -> list[str]:
        if not self.process or not PSUTIL_AVAILABLE:
            return []

        log_files = []
        try:
            for fd in self.process.open_files():
                path = fd.path
                if any(ext in path.lower() for ext in [".log", ".txt", ".out"]):
                    log_files.append(path)
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

        return log_files

    def get_proc_fd_paths(self) -> dict:
        if not self.process or not PSUTIL_AVAILABLE:
            return {}

        fd_paths = {"stdout": None, "stderr": None, "logs": []}

        try:
            pid = self.process.pid
            proc_fd_dir = f"/proc/{pid}/fd"

            if os.path.exists(proc_fd_dir):
                for fd_name in os.listdir(proc_fd_dir):
                    try:
                        fd_path = os.path.join(proc_fd_dir, fd_name)
                        target = os.readlink(fd_path)

                        if fd_name == "1":
                            fd_paths["stdout"] = target
                        elif fd_name == "2":
                            fd_paths["stderr"] = target
                        elif ".log" in target.lower() or ".txt" in target.lower():
                            fd_paths["logs"].append(target)
                    except (OSError, PermissionError):
                        continue
        except (OSError, PermissionError):
            pass

        return fd_paths


class TensorBoardReader:
    """Read metrics from TensorBoard event files."""

    def __init__(self):
        self.event_file = None
        self.last_position = 0
        self.metrics = {}

    def find_event_file(self, search_dir: Path) -> Optional[Path]:
        if not search_dir or not search_dir.exists():
            return None

        patterns = ["events.out.tfevents.*", "events.*"]
        event_files = []

        for pattern in patterns:
            event_files.extend(search_dir.rglob(pattern))

        if event_files:
            return max(
                event_files, key=lambda p: p.stat().st_mtime if p.exists() else 0
            )
        return None

    def read_events(self, event_path: Path) -> dict:
        if not event_path or not event_path.exists():
            return {}

        metrics = {}
        try:
            with open(event_path, "rb") as f:
                f.seek(self.last_position)

                while True:
                    header = f.read(12)
                    if len(header) < 12:
                        break

                    length = struct.unpack("<Q", header[:8])[0]
                    crc = struct.unpack("<I", header[8:12])[0]

                    data = f.read(length)
                    if len(data) < length:
                        break

                    f.read(4)

                    try:
                        self._parse_record(data, metrics)
                    except Exception:
                        pass

                    self.last_position = f.tell()
        except (IOError, struct.error):
            pass

        self.metrics.update(metrics)
        return metrics

    def _parse_record(self, data: bytes, metrics: dict):
        try:
            import tensorflow as tf
            from tensorflow.core.util import event_pb2

            event = event_pb2.Event()
            event.ParseFromString(data)

            if event.HasField("summary"):
                for value in event.summary.value:
                    tag = value.tag
                    if value.HasField("simple_value"):
                        metrics[tag] = value.simple_value
                    elif value.HasField("tensor"):
                        tensor = value.tensor
                        if tensor.float_val:
                            metrics[tag] = tensor.float_val[0]
        except ImportError:
            self._parse_record_manual(data, metrics)

    def _parse_record_manual(self, data: bytes, metrics: dict):
        tag_match = re.search(rb"tag\x00([^\x00]+)", data)
        if tag_match:
            tag = tag_match.group(1).decode("utf-8", errors="ignore")

            float_patterns = [
                rb"simple_value.*?([\x00-\xff]{8})",
                rb"scalar.*?([\x00-\xff]{8})",
            ]

            for pattern in float_patterns:
                match = re.search(pattern, data, re.DOTALL)
                if match:
                    try:
                        value = struct.unpack("<d", match.group(1))[0]
                        if not (value != value):
                            metrics[tag] = value
                            break
                    except struct.error:
                        pass


class TrainingMonitor:
    def __init__(
        self,
        output_dir: Optional[str] = None,
        log_file: Optional[str] = None,
        pid: Optional[int] = None,
    ):
        self.output_dir = Path(output_dir) if output_dir else None
        self.log_file = Path(log_file) if log_file else None
        self.metrics = TrainingMetrics()
        self.last_log_position = 0
        self.last_trainer_state_mtime = 0
        self.max_history = 100
        self.status = MonitorStatus()
        self.process_monitor = ProcessMonitor()
        self.tensorboard_reader = TensorBoardReader()

        if pid:
            self.process_monitor.attach_to_process(pid)

        if not self.output_dir or not self.log_file:
            self.auto_discover()

    def find_training_process(self) -> Optional[int]:
        processes = self.process_monitor.find_training_processes()
        if processes:
            return processes[0].pid
        return None

    def attach_to_training(self, pid: Optional[int] = None) -> bool:
        if pid:
            if self.process_monitor.attach_to_process(pid):
                self._update_status_live()
                return True
            return False

        found_pid = self.find_training_process()
        if found_pid:
            return self.attach_to_training(found_pid)
        return False

    def _update_status_live(self):
        self.status.is_live = self.process_monitor.process is not None
        self.status.source_type = "live_process" if self.status.is_live else "log_file"

        if self.status.is_live and self.process_monitor.training_pid:
            self.status.process_pid = self.process_monitor.training_pid
            proc_info = self.process_monitor.get_process_info()
            if proc_info:
                self.status.process_command = proc_info.command[:80]

            fd_paths = self.process_monitor.get_proc_fd_paths()
            if fd_paths.get("logs"):
                self.status.log_file = fd_paths["logs"][0]
                if not self.log_file:
                    self.log_file = Path(fd_paths["logs"][0])

        if self.output_dir:
            self.status.source_path = str(self.output_dir)
            tb_file = self.tensorboard_reader.find_event_file(self.output_dir)
            if tb_file:
                self.status.tensorboard_dir = str(tb_file.parent)

    def find_trainer_state(self) -> Optional[Path]:
        if not self.output_dir or not self.output_dir.exists():
            return None

        trainer_state = self.output_dir / "trainer_state.json"
        if trainer_state.exists():
            return trainer_state

        checkpoint_dirs = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: (
                int(p.name.split("-")[-1]) if p.name.split("-")[-1].isdigit() else 0
            ),
            reverse=True,
        )

        for ckpt_dir in checkpoint_dirs:
            state_file = ckpt_dir / "trainer_state.json"
            if state_file.exists():
                return state_file

        return None

    def parse_trainer_state(self, state_path: Path) -> dict:
        try:
            with open(state_path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    def update_from_trainer_state(self):
        state_path = self.find_trainer_state()
        if not state_path:
            return

        try:
            mtime = state_path.stat().st_mtime
            if mtime <= self.last_trainer_state_mtime:
                return
            self.last_trainer_state_mtime = mtime
        except OSError:
            return

        state = self.parse_trainer_state(state_path)
        if not state:
            return

        self.metrics.epoch = state.get("epoch")
        self.metrics.current_step = state.get("global_step")
        self.metrics.total_steps = state.get("max_steps")

        log_history = state.get("log_history", [])
        if log_history:
            self.metrics.loss_history = []
            for entry in log_history:
                if "loss" in entry:
                    self.metrics.loss_history.append(
                        {
                            "step": entry.get("step", entry.get("epoch", 0)),
                            "loss": entry["loss"],
                            "learning_rate": entry.get("learning_rate"),
                            "grad_norm": entry.get("grad_norm"),
                        }
                    )

            self.metrics.loss_history = self.metrics.loss_history[-self.max_history :]

            last_entry = log_history[-1]
            if "loss" in last_entry:
                self.metrics.loss = last_entry["loss"]
            if "learning_rate" in last_entry:
                self.metrics.learning_rate = last_entry["learning_rate"]
            if "grad_norm" in last_entry:
                self.metrics.grad_norm = last_entry["grad_norm"]

        self.metrics.last_update = datetime.now()

    def find_log_file(self) -> Optional[Path]:
        if self.log_file and self.log_file.exists():
            return self.log_file

        if self.status.is_live:
            open_logs = self.process_monitor.get_open_log_files()
            if open_logs:
                return Path(open_logs[0])

        search_dirs = []
        if self.output_dir and self.output_dir.exists():
            search_dirs.append(self.output_dir)

        current_dir = Path(".")
        home_dir = Path.home()

        search_dirs.append(current_dir)
        if home_dir.exists():
            search_dirs.append(home_dir)

        patterns = ["train*.log", "*.log", "training*.log", "output*.log"]

        for search_dir in search_dirs:
            for pattern in patterns:
                matches = list(search_dir.glob(pattern))
                if matches:
                    return max(
                        matches, key=lambda p: p.stat().st_mtime if p.exists() else 0
                    )

        return None

    def parse_log_line(self, line: str) -> dict:
        result = {}

        loss_patterns = [
            r"loss[=:]\s*([0-9.]+)",
            r"'loss':\s*([0-9.]+)",
            r"Loss:\s*([0-9.]+)",
        ]
        for pattern in loss_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result["loss"] = float(match.group(1))
                break

        lr_patterns = [
            r"learning_rate[=:]\s*([0-9.eE+-]+)",
            r"'learning_rate':\s*([0-9.eE+-]+)",
            r"lr[=:]\s*([0-9.eE+-]+)",
        ]
        for pattern in lr_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result["learning_rate"] = float(match.group(1))
                break

        step_patterns = [
            r"step[=:]\s*([0-9]+)",
            r"'step':\s*([0-9]+)",
            r"\[([0-9]+)/([0-9]+)\]",
        ]
        for pattern in step_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    result["current_step"] = int(match.group(1))
                    result["total_steps"] = int(match.group(2))
                else:
                    result["step"] = int(match.group(1))
                break

        grad_patterns = [
            r"grad_norm[=:]\s*([0-9.]+)",
            r"'grad_norm':\s*([0-9.]+)",
            r"gradient norm[=:]\s*([0-9.]+)",
        ]
        for pattern in grad_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result["grad_norm"] = float(match.group(1))
                break

        epoch_patterns = [
            r"epoch[=:]\s*([0-9.]+)",
            r"'epoch':\s*([0-9.]+)",
        ]
        for pattern in epoch_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result["epoch"] = float(match.group(1))
                break

        speed_patterns = [
            r"(\d+\.?\d*)\s*samples.?/?\s*s",
            r"(\d+\.?\d*)\s*it/s",
            r"throughput[=:]\s*([0-9.]+)",
        ]
        for pattern in speed_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                result["samples_per_second"] = float(match.group(1))
                break

        return result

    def update_from_log(self, read_all: bool = False):
        log_path = self.find_log_file()
        if not log_path:
            return

        try:
            with open(log_path, "r") as f:
                if read_all:
                    self.last_log_position = 0
                    f.seek(0)
                else:
                    f.seek(self.last_log_position)
                new_lines = f.readlines()
                self.last_log_position = f.tell()

            for line in new_lines:
                parsed = self.parse_log_line(line)
                if parsed:
                    if "loss" in parsed:
                        self.metrics.loss = parsed["loss"]
                        step = parsed.get(
                            "step",
                            parsed.get("current_step", len(self.metrics.loss_history)),
                        )
                        self.metrics.loss_history.append(
                            {
                                "step": step,
                                "loss": parsed["loss"],
                                "learning_rate": parsed.get("learning_rate"),
                                "grad_norm": parsed.get("grad_norm"),
                            }
                        )
                    if "learning_rate" in parsed:
                        self.metrics.learning_rate = parsed["learning_rate"]
                    if "grad_norm" in parsed:
                        self.metrics.grad_norm = parsed["grad_norm"]
                    if "epoch" in parsed:
                        self.metrics.epoch = parsed["epoch"]
                    if "current_step" in parsed:
                        self.metrics.current_step = parsed["current_step"]
                    if "total_steps" in parsed:
                        self.metrics.total_steps = parsed["total_steps"]
                    if "samples_per_second" in parsed:
                        self.metrics.samples_per_second = parsed["samples_per_second"]

            self.metrics.loss_history = self.metrics.loss_history[-self.max_history :]

            if new_lines:
                self.metrics.last_update = datetime.now()

        except IOError:
            pass

    def update_from_tensorboard(self):
        if not self.output_dir:
            return

        event_file = self.tensorboard_reader.find_event_file(self.output_dir)
        if event_file:
            tb_metrics = self.tensorboard_reader.read_events(event_file)

            for tag, value in tb_metrics.items():
                tag_lower = tag.lower()
                if "loss" in tag_lower and "loss" not in self.metrics.__dict__:
                    self.metrics.loss = value
                elif "lr" in tag_lower or "learning_rate" in tag_lower:
                    self.metrics.learning_rate = value
                elif "grad_norm" in tag_lower:
                    self.metrics.grad_norm = value
                elif "epoch" in tag_lower:
                    self.metrics.epoch = value

    def update(self):
        if self.output_dir is None and self.log_file is None:
            self.auto_discover()

        if not self.status.is_live:
            self.attach_to_training()

        if self.status.is_live:
            self._update_status_live()

        self.update_from_trainer_state()
        self.update_from_log()
        self.update_from_tensorboard()

    def reset_log_position(self):
        self.last_log_position = 0

    def auto_discover(self):
        current_dir = Path(".")
        home_dir = Path.home()

        if not self.output_dir:
            output_patterns = [
                "*-output",
                "*output",
                "outputs",
                "output",
                "runs",
                "checkpoints",
            ]
            output_dirs = []
            for pattern in output_patterns:
                output_dirs.extend(current_dir.glob(pattern))
                output_dirs.extend(current_dir.glob(f"{pattern}/*"))

            if output_dirs:
                self.output_dir = max(
                    output_dirs,
                    key=lambda p: p.stat().st_mtime if p.exists() and p.is_dir() else 0,
                )

        if not self.log_file:
            log_patterns = ["train*.log", "*.log"]
            log_files = []

            for pattern in log_patterns:
                log_files.extend(current_dir.glob(pattern))

            recent_logs = sorted(
                log_files,
                key=lambda p: p.stat().st_mtime if p.exists() else 0,
                reverse=True,
            )

            if recent_logs:
                self.log_file = recent_logs[0]

        self.attach_to_training()


class MonitorUI:
    def __init__(self, gpu_monitor: GPUMonitor, training_monitor: TrainingMonitor):
        self.gpu_monitor = gpu_monitor
        self.training_monitor = training_monitor
        self.console = Console()
        self.layout = Layout()
        self._setup_layout()

    def _setup_layout(self):
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="status", size=4),
            Layout(name="gpu", size=12),
            Layout(name="training", size=10),
            Layout(name="process", size=6),
            Layout(name="loss_chart", size=12),
            Layout(name="footer", size=2),
        )

    def _create_status_panel(self) -> Panel:
        status = self.training_monitor.status

        if status.is_live:
            status_text = Text()
            status_text.append("● ", style="bold green")
            status_text.append("LIVE TRAINING", style="bold green")
            status_text.append("\n")

            if status.process_pid:
                status_text.append(f"PID: {status.process_pid}", style="cyan")
                status_text.append(" | ")

            if status.process_command:
                cmd_short = (
                    status.process_command[:60] + "..."
                    if len(status.process_command) > 60
                    else status.process_command
                )
                status_text.append(f"CMD: {cmd_short}", style="yellow")
        else:
            status_text = Text()
            status_text.append("○ ", style="yellow")
            status_text.append("MONITORING LOGS", style="yellow")
            status_text.append(" (no active training process)", style="dim")
            status_text.append("\n")

            if status.source_path:
                status_text.append(f"Source: {status.source_path}", style="dim")

        return Panel(
            status_text,
            title="Monitor Status",
            border_style="green" if status.is_live else "yellow",
        )

    def _create_process_table(self, proc_info: Optional[ProcessInfo]) -> Table:
        table = Table(
            title="Process Metrics",
            expand=True,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Metric", style="magenta", width=15)
        table.add_column("Value", justify="right", width=18)
        table.add_column("Metric", style="magenta", width=15)
        table.add_column("Value", justify="right", width=18)

        if proc_info:
            cpu_str = f"{proc_info.cpu_percent:.1f}%"
            mem_str = f"{proc_info.memory_mb:.0f}MB ({proc_info.memory_percent:.1f}%)"
            read_str = f"{proc_info.read_mb:.1f}MB"
            write_str = f"{proc_info.write_mb:.1f}MB"
            status_str = proc_info.status
            pid_str = str(proc_info.pid)
        else:
            cpu_str = mem_str = read_str = write_str = status_str = pid_str = "N/A"

        table.add_row("CPU", cpu_str, "Memory", mem_str)
        table.add_row("Disk Read", read_str, "Disk Write", write_str)
        table.add_row("PID", pid_str, "Status", status_str)

        return table

    def _create_gpu_table(self, gpu_infos: list[GPUInfo]) -> Table:
        table = Table(
            title="GPU Status", expand=True, show_header=True, header_style="bold cyan"
        )

        table.add_column("GPU", style="cyan", width=6)
        table.add_column("Util%", justify="right", width=6)
        table.add_column("Memory", justify="right", width=14)
        table.add_column("Temp", justify="right", width=6)
        table.add_column("Power", justify="right", width=12)
        table.add_column("Clock", justify="right", width=14)
        table.add_column("Fan%", justify="right", width=5)
        table.add_column("Procs", justify="right", width=5)

        for info in gpu_infos:
            util_color = self._get_color_for_percent(info.utilization)
            mem_color = self._get_color_for_percent(info.memory_percent)
            temp_color = self._get_color_for_temp(info.temperature)
            power_color = self._get_color_for_percent(info.power_percent)

            mem_str = f"{info.memory_used:.1f}/{info.memory_total:.1f}GB"
            power_str = f"{info.power_usage:.0f}/{info.power_limit:.0f}W"
            clock_str = f"{info.clock_core}/{info.clock_memory}MHz"

            table.add_row(
                f"[bold]{info.index}[/bold]",
                f"[{util_color}]{info.utilization:.0f}%[/{util_color}]",
                f"[{mem_color}]{mem_str}[/{mem_color}]",
                f"[{temp_color}]{info.temperature}°C[/{temp_color}]",
                f"[{power_color}]{power_str}[/{power_color}]",
                clock_str,
                f"{info.fan_speed}%",
                str(len(info.processes)),
            )

        return table

    def _create_training_table(self, metrics: TrainingMetrics) -> Table:
        table = Table(
            title="Training Metrics",
            expand=True,
            show_header=True,
            header_style="bold green",
        )

        table.add_column("Metric", style="green", width=15)
        table.add_column("Value", justify="right", width=20)
        table.add_column("Metric", style="green", width=15)
        table.add_column("Value", justify="right", width=20)

        loss_str = f"{metrics.loss:.6f}" if metrics.loss is not None else "N/A"
        lr_str = (
            f"{metrics.learning_rate:.2e}"
            if metrics.learning_rate is not None
            else "N/A"
        )
        epoch_str = f"{metrics.epoch:.2f}" if metrics.epoch is not None else "N/A"
        grad_str = (
            f"{metrics.grad_norm:.4f}" if metrics.grad_norm is not None else "N/A"
        )
        speed_str = (
            f"{metrics.samples_per_second:.2f}"
            if metrics.samples_per_second is not None
            else "N/A"
        )

        step_str = "N/A"
        if metrics.current_step is not None:
            if metrics.total_steps is not None:
                progress = (
                    (metrics.current_step / metrics.total_steps) * 100
                    if metrics.total_steps > 0
                    else 0
                )
                step_str = (
                    f"{metrics.current_step}/{metrics.total_steps} ({progress:.1f}%)"
                )
            else:
                step_str = str(metrics.current_step)

        history_str = str(len(metrics.loss_history))
        update_str = (
            metrics.last_update.strftime("%H:%M:%S") if metrics.last_update else "N/A"
        )

        table.add_row(
            "Loss",
            loss_str,
            "Learning Rate",
            lr_str,
        )
        table.add_row(
            "Epoch",
            epoch_str,
            "Grad Norm",
            grad_str,
        )
        table.add_row(
            "Step",
            step_str,
            "Speed",
            speed_str,
        )
        table.add_row(
            "Last Update",
            update_str,
            "History Points",
            history_str,
        )

        return table

    def _create_loss_chart(
        self, metrics: TrainingMetrics, width: int = 80, height: int = 8
    ) -> str:
        if not metrics.loss_history:
            return "No loss history available"

        losses = [h["loss"] for h in metrics.loss_history if h.get("loss") is not None]
        if not losses:
            return "No valid loss values"

        min_loss = min(losses)
        max_loss = max(losses)

        if max_loss == min_loss:
            max_loss = min_loss + 1

        chart_lines = []
        chart_lines.append(
            f"Loss Range: [{min_loss:.4f}, {max_loss:.4f}] | Points: {len(losses)}"
        )
        chart_lines.append("─" * min(width, 100))

        chart_height = height - 3
        chart_width = min(len(losses), width - 10)

        step_size = max(1, len(losses) // chart_width) if chart_width > 0 else 1
        sampled_losses = losses[::step_size][:chart_width]

        grid = [[" " for _ in range(len(sampled_losses))] for _ in range(chart_height)]

        for i, loss in enumerate(sampled_losses):
            normalized = (loss - min_loss) / (max_loss - min_loss)
            y = int((1 - normalized) * (chart_height - 1))
            y = max(0, min(chart_height - 1, y))
            grid[y][i] = "●"

        y_labels = []
        for i in range(chart_height):
            val = (
                max_loss - (i / (chart_height - 1)) * (max_loss - min_loss)
                if chart_height > 1
                else max_loss
            )
            y_labels.append(f"{val:6.3f}│")

        for i, (label, row) in enumerate(zip(y_labels, grid)):
            line = label + "".join(row)
            color = (
                "green"
                if i > chart_height // 2
                else "yellow"
                if i > chart_height // 4
                else "red"
            )
            chart_lines.append(f"[{color}]{line}[/{color}]")

        chart_lines.append("       └" + "─" * len(sampled_losses))
        chart_lines.append("        " + "Steps →")

        return Text.from_markup("\n".join(chart_lines))

    def _get_color_for_percent(self, percent: float) -> str:
        if percent < 50:
            return "green"
        elif percent < 80:
            return "yellow"
        else:
            return "red"

    def _get_color_for_temp(self, temp: int) -> str:
        if temp < 60:
            return "green"
        elif temp < 80:
            return "yellow"
        else:
            return "red"

    def _render_header(self) -> Panel:
        title = Text("LLM Training Monitor", style="bold magenta")
        subtitle = Text(
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style="dim"
        )
        content = Text.assemble(title, " | ", subtitle)
        return Panel(content, style="bold blue")

    def _render_footer(self) -> Panel:
        controls = "Controls: Ctrl+C to exit | --pid PID to attach to process"
        if not NVML_AVAILABLE and not NVIDIA_SMI_AVAILABLE:
            controls += " | [yellow]No GPU monitoring available (pynvml and nvidia-smi missing)[/yellow]"
        elif not NVML_AVAILABLE and NVIDIA_SMI_AVAILABLE:
            controls += (
                " | [yellow]Using nvidia-smi fallback for GPU monitoring[/yellow]"
            )
        return Panel(controls, style="dim")

    def update_display(self) -> Layout:
        gpu_infos = self.gpu_monitor.get_all_gpu_info()
        metrics = self.training_monitor.metrics
        proc_info = self.training_monitor.process_monitor.get_process_info()

        self.layout["header"].update(self._render_header())
        self.layout["status"].update(self._create_status_panel())
        self.layout["gpu"].update(
            Panel(self._create_gpu_table(gpu_infos), border_style="cyan")
        )
        self.layout["training"].update(
            Panel(self._create_training_table(metrics), border_style="green")
        )
        self.layout["process"].update(
            Panel(self._create_process_table(proc_info), border_style="magenta")
        )
        self.layout["loss_chart"].update(
            Panel(
                self._create_loss_chart(metrics),
                title="Loss History",
                border_style="yellow",
            )
        )
        self.layout["footer"].update(self._render_footer())

        return self.layout

    def render_once(self):
        self.training_monitor.update()
        self.console.print(self._render_header())
        self.console.print(self._create_status_panel())
        self.console.print(self._create_gpu_table(self.gpu_monitor.get_all_gpu_info()))
        self.console.print(self._create_training_table(self.training_monitor.metrics))
        proc_info = self.training_monitor.process_monitor.get_process_info()
        self.console.print(self._create_process_table(proc_info))
        self.console.print(
            Panel(
                self._create_loss_chart(self.training_monitor.metrics),
                title="Loss History",
                border_style="yellow",
            )
        )
        self.console.print(self._render_footer())

    def run_live(self, interval: float = 1.0):
        try:
            with Live(
                self._create_live_display(),
                console=self.console,
                refresh_per_second=1 / interval,
            ) as live:
                while True:
                    self.training_monitor.update()
                    live.update(self._create_live_display())
                    time.sleep(interval)
        except KeyboardInterrupt:
            self.console.print("\n[yellow]Monitor stopped by user.[/yellow]")

    def _create_live_display(self):
        from rich.console import Group

        proc_info = self.training_monitor.process_monitor.get_process_info()
        return Group(
            self._render_header(),
            self._create_status_panel(),
            self._create_gpu_table(self.gpu_monitor.get_all_gpu_info()),
            self._create_training_table(self.training_monitor.metrics),
            self._create_process_table(proc_info),
            Panel(
                self._create_loss_chart(self.training_monitor.metrics),
                title="Loss History",
                border_style="yellow",
            ),
            self._render_footer(),
        )


def list_training_processes():
    if not PSUTIL_AVAILABLE:
        print("Error: psutil not available. Install with: pip install psutil")
        return

    monitor = ProcessMonitor()
    processes = monitor.find_training_processes()

    if not processes:
        print("No training processes found.")
        return

    print(f"\nFound {len(processes)} training process(es):\n")
    print(f"{'PID':<8} {'CPU%':<8} {'Memory':<15} {'Command'}")
    print("-" * 80)

    for proc in processes:
        cmd_short = (
            proc.command[:60] + "..." if len(proc.command) > 60 else proc.command
        )
        print(
            f"{proc.pid:<8} {proc.cpu_percent:<8.1f} {proc.memory_mb:<15.0f} {cmd_short}"
        )


def print_single_report(args):
    gpu_monitor = GPUMonitor()
    training_monitor = TrainingMonitor(args.output_dir, args.log_file, args.pid)
    ui = MonitorUI(gpu_monitor, training_monitor)

    training_monitor.update()
    ui.render_once()

    gpu_monitor.shutdown()


def run_continuous_monitor(args):
    gpu_monitor = GPUMonitor()
    training_monitor = TrainingMonitor(args.output_dir, args.log_file, args.pid)
    ui = MonitorUI(gpu_monitor, training_monitor)

    try:
        ui.run_live(interval=args.interval)
    finally:
        gpu_monitor.shutdown()


def main():
    parser = argparse.ArgumentParser(
        description="Monitor GPU usage and LLM training progress",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --watch                          # Auto-detect and monitor training
  %(prog)s --pid 12345 --watch              # Attach to specific process
  %(prog)s --output-dir ./outputs --watch   # Monitor specific output directory
  %(prog)s --log-file training.log          # Monitor specific log file
  %(prog)s --list                           # List all training processes
  %(prog)s --once                           # Print single report and exit
        """,
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default=None,
        help="Directory containing training outputs (checkpoints, logs)",
    )

    parser.add_argument(
        "--log-file", "-l", type=str, default=None, help="Specific log file to monitor"
    )

    parser.add_argument(
        "--pid",
        "-p",
        type=int,
        default=None,
        help="PID of training process to attach to",
    )

    parser.add_argument(
        "--interval",
        "-i",
        type=float,
        default=1.0,
        help="Update interval in seconds (default: 1.0)",
    )

    parser.add_argument(
        "--watch", "-w", action="store_true", help="Run continuous monitoring"
    )

    parser.add_argument(
        "--once", action="store_true", help="Print single report and exit"
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List all detected training processes",
    )

    args = parser.parse_args()

    if args.list:
        list_training_processes()
        return

    if args.output_dir is None and args.log_file is None and args.pid is None:
        print(
            "Auto-discover enabled: searching for training processes, "
            "output directories and log files...",
            file=sys.stderr,
        )

    if not PSUTIL_AVAILABLE:
        print(
            "Warning: psutil not available. Process monitoring disabled. "
            "Install with: pip install psutil",
            file=sys.stderr,
        )

    if not NVML_AVAILABLE and not NVIDIA_SMI_AVAILABLE:
        print(
            "Warning: pynvml and nvidia-smi not available. GPU monitoring disabled.",
            file=sys.stderr,
        )
    elif not NVML_AVAILABLE and NVIDIA_SMI_AVAILABLE:
        print(
            "Note: pynvml not available, using nvidia-smi fallback for GPU monitoring.",
            file=sys.stderr,
        )

    if args.once:
        print_single_report(args)
    else:
        run_continuous_monitor(args)


if __name__ == "__main__":
    main()
