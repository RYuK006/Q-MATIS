import threading
import time
import datetime
import json
import logging
import psutil
import torch
from superconductor.materials_lake import MaterialsLake
from superconductor.experiment import Experiment

logger = logging.getLogger(__name__)

class ResourceMonitor:
    def __init__(self, experiment: Experiment, lake: MaterialsLake, interval_seconds: int = 60):
        self.experiment = experiment
        self.lake = lake
        self.interval = interval_seconds
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        if self._thread is None or not self._thread.is_alive():
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._thread.start()
            logger.info("ResourceMonitor background thread started.")

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._stop_event.set()
            self._thread.join(timeout=self.interval + 2)
            logger.info("ResourceMonitor background thread stopped.")

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            try:
                self._record_metrics()
            except Exception as e:
                logger.error(f"Failed to record resource metrics: {e}")
            
            # Wait for interval or until stopped
            self._stop_event.wait(self.interval)

    def _record_metrics(self):
        # CPU & RAM
        cpu_usage = psutil.cpu_percent(interval=None)
        ram_info = psutil.virtual_memory()
        ram_usage_mb = ram_info.used / (1024 * 1024)
        
        # GPU & VRAM (using PyTorch)
        gpu_usage = 0.0
        vram_usage_mb = 0.0
        if torch.cuda.is_available():
            try:
                # Approximate GPU usage could be hard without pynvml, using vram as a proxy for now
                vram_allocated = torch.cuda.memory_allocated() / (1024 * 1024)
                vram_usage_mb = vram_allocated
            except Exception:
                pass
                
        # Disk usage of DB
        db_size_mb = 0.0
        try:
            db_size_mb = os.path.getsize(self.lake.db_path) / (1024 * 1024)
        except Exception:
            pass
            
        disk_usage_mb = psutil.disk_usage('/').used / (1024 * 1024)
        
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        
        self.lake.execute_write("""
            INSERT INTO resource_history (
                experiment_id, cpu_usage_percent, ram_usage_mb, 
                gpu_usage_percent, vram_usage_mb, disk_usage_mb, 
                db_size_mb, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            self.experiment.id, cpu_usage, ram_usage_mb,
            gpu_usage, vram_usage_mb, disk_usage_mb,
            db_size_mb, now
        ))
