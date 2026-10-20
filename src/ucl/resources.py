"""Small local resource guard; no system configuration changes."""
from __future__ import annotations
import shutil
import subprocess
import time
import psutil


def snapshot():
    memory = psutil.virtual_memory()
    value = {"ram_available_mib": memory.available / 2**20,
             "process_rss_mib": psutil.Process().memory_info().rss / 2**20,
             "process_cpu_seconds": sum(psutil.Process().cpu_times()[:2])}
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3, check=True)
        temp, used, total, utilization = map(float, result.stdout.splitlines()[0].split(","))
        value.update(gpu_temperature_c=temp, gpu_used_mib=used, gpu_total_mib=total,
                     gpu_utilization_percent=utilization)
    except (OSError, ValueError, subprocess.SubprocessError):
        value["gpu_measurement_available"] = False
    return value


class ResourceGuard:
    def __init__(self, limits=None, directory="."):
        self.limits = {"min_ram_mib": 400, "max_temperature_c": 82, "min_disk_gib": 2,
                       "max_seconds": 600, **(limits or {})}
        self.directory, self.started = directory, time.monotonic()
        self.samples = []

    def check(self):
        result = snapshot()
        result["elapsed_seconds"] = time.monotonic() - self.started
        self.samples.append(result)
        if result["ram_available_mib"] < self.limits["min_ram_mib"]:
            raise RuntimeError("Resource limit: available RAM below configured minimum")
        if result.get("gpu_temperature_c", 0) >= self.limits["max_temperature_c"]:
            raise RuntimeError("Resource limit: GPU temperature")
        if shutil.disk_usage(self.directory).free < self.limits["min_disk_gib"] * 2**30:
            raise RuntimeError("Resource limit: insufficient disk space")
        if result["elapsed_seconds"] > self.limits["max_seconds"]:
            raise RuntimeError("Resource limit: execution time")
        return result
