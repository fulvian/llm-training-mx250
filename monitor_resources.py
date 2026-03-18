#!/usr/bin/env python3
"""Monitor GPU + CPU per training LLM"""

import subprocess
import time
import os

def get_gpu_stats():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            return {
                'util': int(parts[0]),
                'mem': int(parts[1]),
                'total': int(parts[2]),
                'temp': int(parts[3]),
            }
    except: pass
    return None

def find_training():
    for d in os.listdir('/proc'):
        if not d.isdigit(): continue
        try:
            with open(f'/proc/{d}/cmdline') as f:
                cmd = f.read().lower()
                if 'python' in cmd and 'train' in cmd:
                    return int(d)
        except: pass
    return None

def get_proc_stats(pid):
    try:
        with open(f'/proc/{pid}/stat') as f:
            parts = f.read().split()
            cpu = int(parts[13]) + int(parts[14])
            rss = int(parts[23]) * 4  # pages to KB
        return {'cpu': cpu, 'rss': rss}
    except: pass
    return {'cpu': 0, 'rss': 0}

def get_total_cpu():
    try:
        with open('/proc/stat') as f:
            line = f.readline()
            parts = line.split()
            return sum(int(x) for x in parts[1:8])
    except: pass
    return 1

def main():
    pid = find_training()
    
    print("=" * 70)
    print("GPU + CPU Training Monitor")
    print("=" * 70)
    if pid:
        print(f"Training PID: {pid}")
    else:
        print("Nessun training trovato - avvia in un altro terminale")
    print("=" * 70)
    print()
    
    prev_cpu = 0
    prev_total = 0
    n_cpus = os.cpu_count() or 1
    
    try:
        while True:
            gpu = get_gpu_stats()
            
            if not pid:
                pid = find_training()
            
            if pid:
                proc = get_proc_stats(pid)
                total = get_total_cpu()
                
                d_cpu = proc['cpu'] - prev_cpu
                d_total = total - prev_total if total != prev_total else 1
                cpu_pct = min(d_cpu / d_total * 100 * n_cpus, 100 * n_cpus)
                
                prev_cpu = proc['cpu']
                prev_total = total
            else:
                cpu_pct = 0
                proc = {'rss': 0}
            
            if gpu:
                mem_pct = gpu['mem'] / gpu['total'] * 100
                print(f"\rGPU: {gpu['util']:3d}% | MEM: {gpu['mem']:4d}/{gpu['total']}MB ({mem_pct:5.1f}%) | "
                      f"TEMP: {gpu['temp']}°C | CPU: {cpu_pct:5.1f}% | RAM: {proc['rss']//1024:4d}MB", end="", flush=True)
            
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\nMonitor fermato.")

if __name__ == "__main__":
    main()
