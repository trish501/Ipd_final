import os
import glob
import json
import subprocess
import threading
import psutil
from datetime import datetime
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline_process = None
pipeline_start_time = None
pipeline_config = None

class PipelineConfig(BaseModel):
    source: str = "VIIRS"
    loc_type: str = "World"
    loc_val: str = "World"
    bbox: Optional[str] = None
    sat: str = "Sentinel-2"
    mode: str = "BASELINE_B4_B11_B12"
    start_date: str = "01-01-2025"
    end_date: str = "28-02-2025"
    target_images: int = 10

def run_pipeline(config: PipelineConfig):
    global pipeline_process, pipeline_start_time, pipeline_config
    
    cmd = [
        "python", "main.py",
        "--source", config.source,
        "--loc-type", config.loc_type,
        "--loc-val", config.loc_val,
        "--sat", config.sat,
        "--mode", config.mode,
        "--start-date", config.start_date,
        "--end-date", config.end_date,
        "--target-images", str(config.target_images),
        "--max-workers", "20"
    ]
    
    if config.bbox:
        cmd.extend(["--bbox", config.bbox])
        
    pipeline_start_time = datetime.now()
    pipeline_config = config
    
    import sys
    pipeline_process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(os.path.abspath(__file__)),
        stdout=sys.stdout,
        stderr=sys.stderr
    )
    
    pipeline_process.wait()

@app.post("/api/start")
def start_pipeline(config: PipelineConfig, background_tasks: BackgroundTasks):
    global pipeline_process
    if pipeline_process and pipeline_process.poll() is None:
        raise HTTPException(status_code=400, detail="Pipeline is already running.")
        
    background_tasks.add_task(run_pipeline, config)
    return {"message": "Pipeline started"}

@app.post("/api/stop")
def stop_pipeline():
    global pipeline_process
    if pipeline_process and pipeline_process.poll() is None:
        pipeline_process.terminate()
        return {"message": "Pipeline stopped"}
    return {"message": "Pipeline is not running"}

@app.get("/api/status")
def get_status():
    global pipeline_process, pipeline_start_time, pipeline_config
    
    # 1. Check for any running main.py process
    is_running = False
    external_start_time = None
    active_config = None
    
    # Check our spawned process first
    if pipeline_process and pipeline_process.poll() is None:
        is_running = True
        external_start_time = pipeline_start_time.timestamp() if pipeline_start_time else None
        if pipeline_config:
            active_config = {
                "start_date": pipeline_config.start_date,
                "end_date": pipeline_config.end_date,
                "target_images": pipeline_config.target_images
            }
    else:
        # Check system processes for python main.py
        for p in psutil.process_iter(['name', 'cmdline', 'create_time']):
            try:
                cmd = p.info.get('cmdline') or []
                if 'python' in p.info.get('name', '').lower() or any('python' in c.lower() for c in cmd):
                    if any('main.py' in c for c in cmd) and 'api.py' not in cmd:
                        is_running = True
                        external_start_time = p.info.get('create_time')
                        
                        active_config = {
                            "start_date": "01-01-2025",
                            "end_date": "31-12-2025",
                            "target_images": 1500
                        }
                        try:
                            if '--start-date' in cmd:
                                active_config["start_date"] = cmd[cmd.index('--start-date') + 1]
                            if '--end-date' in cmd:
                                active_config["end_date"] = cmd[cmd.index('--end-date') + 1]
                            if '--target-images' in cmd:
                                active_config["target_images"] = int(cmd[cmd.index('--target-images') + 1])
                        except Exception:
                            pass
                        
                        break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    
    elapsed = 0
    if is_running and external_start_time:
        elapsed = datetime.now().timestamp() - external_start_time
    elif pipeline_start_time:
        elapsed = (datetime.now() - pipeline_start_time).total_seconds()
        
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_dir = os.path.join(base_dir, "data", "csv")
    metadata_dir = os.path.join(base_dir, "dataset", "metadata")
    state_file = os.path.join(metadata_dir, "pipeline_state.json")
    
    # Calculate Phase 1 progress (CSV downloads)
    expected_csvs = 0
    if pipeline_config:
        try:
            start = datetime.strptime(pipeline_config.start_date, "%d-%m-%Y")
            end = datetime.strptime(pipeline_config.end_date, "%d-%m-%Y")
            expected_csvs = (end - start).days + 1
        except Exception:
            pass
            
    csv_files = []
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv') and not f.startswith('global_')]
    
    csv_count = len(csv_files)
    phase1_progress = min(100.0, (csv_count / expected_csvs * 100) if expected_csvs > 0 else 0)
    
    # Calculate Phase 2 progress (Image generation)
    phase2_progress = 0.0
    generated = 0
    failed = 0
    cached = 0
    total_events = 0
    
    progress_file = os.path.join(metadata_dir, "progress.json")
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                stats = json.load(f)
            total_events = stats.get("total", 0)
            generated = stats.get("downloaded", 0)
            cached = stats.get("cached", 0)
            failed = stats.get("failed", 0) + stats.get("skipped_urban", 0) + stats.get("skipped_no_sat", 0) + stats.get("skipped_black", 0)
            
            target = active_config["target_images"] if active_config else (pipeline_config.target_images if pipeline_config else 1500)
            phase2_progress = min(100.0, ((generated + cached) / target * 100) if target > 0 else 0)
        except Exception:
            pass
    elif os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state_dict = json.load(f)
            total_events = len(state_dict)
            generated = sum(1 for v in state_dict.values() if v == "COMPLETED")
            failed = sum(1 for v in state_dict.values() if v == "FAILED")
            
            target = active_config["target_images"] if active_config else (pipeline_config.target_images if pipeline_config else 1500)
            phase2_progress = min(100.0, (generated / target * 100) if target > 0 else 0)
        except Exception:
            pass
            
    current_phase = "IDLE"
    if is_running:
        if csv_count < expected_csvs and total_events == 0:
            current_phase = "PHASE_1_DOWNLOADING_CSVS"
        else:
            current_phase = "PHASE_2_GENERATING_IMAGES"
            
    overall_progress = 0
    if current_phase == "PHASE_1_DOWNLOADING_CSVS":
        overall_progress = phase1_progress * 0.2 # Phase 1 is 20% of total bar
    elif current_phase == "PHASE_2_GENERATING_IMAGES":
        overall_progress = 20 + (phase2_progress * 0.8) # Phase 2 is 80%
    elif not is_running and generated > 0:
        overall_progress = 100.0
        current_phase = "COMPLETED"

    # ETA Calculation
    eta_seconds = 0
    if is_running and elapsed > 0 and overall_progress > 0:
        total_estimated_time = (elapsed / overall_progress) * 100
        eta_seconds = total_estimated_time - elapsed

    return {
        "is_running": is_running,
        "current_phase": current_phase,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta_seconds,
        "overall_progress": overall_progress,
        "active_config": active_config,
        "phase1": {
            "downloaded": csv_count,
            "expected": expected_csvs,
            "progress": phase1_progress
        },
        "phase2": {
            "generated": generated,
            "cached": cached,
            "failed": failed,
            "total_events_checked": total_events,
            "target": active_config["target_images"] if active_config else (pipeline_config.target_images if pipeline_config else 1500),
            "progress": phase2_progress
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("Backend server running on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
