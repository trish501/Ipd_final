import os
import logging
import argparse
import random
import time
import threading
import json
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import subprocess

# GDAL / Rasterio Optimizations for Cloud Optimized GeoTIFFs (COGs)
# These drastically reduce network roundtrips when downloading small image crops
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["CPL_VSIL_CURL_ALLOWED_EXTENSIONS"] = "tif,tiff"
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "536870912"
os.environ["GDAL_HTTP_MULTIMAC"] = "YES"
os.environ["GDAL_HTTP_MERGE_CONSECUTIVE_RANGES"] = "YES"

from src.fire_data import load_events_from_csv, get_firms_api_key
from src.filters import get_bounding_box
from src.satellite_search import search_satellite_imagery
from src.image_downloader import download_and_crop_image

from src.offline_urban_filter import init_offline_filter, is_in_urban_area
from src.offline_industrial_filter import init_industrial_filter, is_near_industrial
from src.cache import init_cache

logging.basicConfig(level=logging.INFO, format='%(message)s', filename='pipeline.log', filemode='w')
logger = logging.getLogger(__name__)

# Thread-safe resources
csv_lock = threading.Lock()

import sys
import threading
from datetime import datetime

def clear_screen():
    print("\033[2J\033[H", end="")

def run_cli():
    clear_screen()
    print("============================================================")
    print("        URBAN FIRE SATELLITE DATASET PIPELINE")
    print("============================================================")
    print("\n============================================================")
    print("SELECT FIRE DATA SOURCE")
    print("============================================================")
    print("\n1) VIIRS")
    print("   NASA VIIRS active-fire data obtained through FIRMS.\n")
    print("2) MODIS")
    print("   NASA MODIS active-fire data obtained through FIRMS.\n")
    source = ""
    while True:
        c = input("Enter choice: ").strip()
        if c == '1': source = "VIIRS"; break
        if c == '2': source = "MODIS"; break

    clear_screen()
    print("Select Location:\n")
    print("1) World")
    print("   Search fire events globally.\n")
    print("2) Enter location/country")
    print("   Type the country or location you want.\n")
    print("3) Custom")
    print("   Enter a custom geographic area.\n")
    loc_type = ""
    loc_val = ""
    bbox = None
    while True:
        c = input("Enter choice: ").strip()
        if c == '1':
            loc_type = "World"
            loc_val = "World"
            break
        if c == '2':
            loc_type = "Country"
            print("\nEnter country/location:")
            print("Example:\nIndia")
            loc_val = input("").strip()
            pass
            break
        if c == '3':
            loc_type = "Custom"
            print("Enter custom bounding box coordinates:")
            try:
                min_lat = float(input("Min Lat: "))
                max_lat = float(input("Max Lat: "))
                min_lon = float(input("Min Lon: "))
                max_lon = float(input("Max Lon: "))
                bbox = (min_lat, max_lat, min_lon, max_lon)
                loc_val = f"Custom [{min_lat}, {max_lat}, {min_lon}, {max_lon}]"
                break
            except:
                print("Invalid input.")
                continue

    clear_screen()
    print("Select Satellite:\n")
    print("Choose the satellite source used to generate the satellite")
    print("images.\n")
    print("1) Sentinel-2\n")
    sat = ""
    while True:
        c = input("Enter choice: ").strip()
        if c == '1' or c == '': 
            sat = "Sentinel-2"
            break

    clear_screen()
    print("------------------------------------------------------------")
    print("DATE RANGE")
    print("------------------------------------------------------------")
    print("\nEnter the date range for fire data.\n")
    print("Format:\nDD-MM-YYYY\n")
    print("Example:\nStart date: 01-01-2025\nEnd date:   28-02-2025\n")
    print("Only data between these two dates will be requested.")
    print("------------------------------------------------------------\n")
    
    start_dt = None
    end_dt = None
    start_str = ""
    end_str = ""
    while True:
        try:
            start_str = input("Start date: ").strip()
            end_str = input("End date: ").strip()
            
            start_dt = datetime.strptime(start_str, "%d-%m-%Y")
            end_dt = datetime.strptime(end_str, "%d-%m-%Y")
            
            if start_dt > end_dt:
                print("\nInvalid date. Start date must be before or equal to end date.\n")
                continue
                
            break
        except ValueError:
            print("\nInvalid date.\n")
            print("Please use:\nDD-MM-YYYY\n")
            print("Example:\n01-01-2025\n")

    clear_screen()
    print("PIPELINE MODE\n")
    print("1) BASELINE_B4_B11_B12")
    print("   Scientific baseline using only B4, B11, B12.")
    print("2) B8A_AUXILIARY")
    print("   Uses B8A for false-positive suppression.\n")
    mode = "BASELINE_B4_B11_B12"
    while True:
        c = input("Enter choice [1]: ").strip()
        if c == '1' or c == '': 
            mode = "BASELINE_B4_B11_B12"
            break
        elif c == '2':
            mode = "B8A_AUXILIARY"
            break

    clear_screen()
    print("TARGET IMAGES\n")
    print("Enter how many successful fire-event images you want.\n")
    print("Example:\n30\n")
    print("The pipeline will stop after the requested number of successful")
    print("images has been generated.\n")
    target_images = 0
    while True:
        c = input("Target images: ").strip()
        if c.isdigit() and int(c) > 0:
            target_images = int(c)
            break

    clear_screen()
    print("============================================================")
    print("                PIPELINE CONFIGURATION")
    print("============================================================")
    print(f"\nData Source : {source}")
    print(f"Location    : {loc_val}")
    print(f"Start Date  : {start_str}")
    print(f"End Date    : {end_str}")
    print(f"Satellite   : {sat}")
    print(f"Mode        : {mode}")
    print(f"Target      : {target_images} images\n")
    print("============================================================\n")
    print("Starting pipeline...\n")
    time.sleep(2)

    return {
        "source": source,
        "loc_type": loc_type,
        "loc_val": loc_val,
        "bbox": bbox,
        "sat": sat,
        "start_str": start_str,
        "end_str": end_str,
        "start_dt": start_dt,
        "end_dt": end_dt,
        "mode": mode,
        "target_images": target_images
    }

class Dashboard:
    def __init__(self, target_images):
        self.target_images = target_images
        self.status = "Initializing..."
        self.data_source = "FIRMS"
        self.location_str = "World"
        self.date_str = "N/A -> N/A"
        self.images_generated = 0
        self.process = "Starting up..."
        self.lock = threading.Lock()
        self.spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.spinner_idx = 0
        self.lines_drawn = 0

    def update(self, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)

    def render(self):
        with self.lock:
            # Minimal, single-line output using carriage return
            # Clean up newlines from process if any
            clean_process = str(self.process).replace('\n', ' | ')
            out = f"\r[ {self.spinner[self.spinner_idx]} ] Images: {self.images_generated}/{self.target_images} | {clean_process}"
            
            # Pad to ensure previous longer strings are overwritten, but cap at terminal width to avoid line wrapping
            out = out.ljust(100)[:100]
            
            sys.stdout.write(out)
            sys.stdout.flush()
            self.spinner_idx = (self.spinner_idx + 1) % len(self.spinner)
            self.lines_drawn = 1

class ProgressTracker:
    def __init__(self, total, target_images):
        self.total = total
        self.target_images = target_images
        self.processed = 0
        self.downloaded = 0
        self.downloaded_industrial = 0
        self.cached = 0
        self.cached_industrial = 0
        self.failed = 0
        self.skipped_urban = 0
        self.skipped_no_sat = 0
        self.skipped_black = 0
        self.start_time = time.time()
        self.lock = threading.Lock()
        self.stop_requested = False
        
    def add_result(self, result_type):
        with self.lock:
            self.processed += 1
            if result_type == "downloaded":
                self.downloaded += 1
            elif result_type == "downloaded_industrial":
                self.downloaded_industrial += 1
            elif result_type == "cached":
                self.cached += 1
            elif result_type == "cached_industrial":
                self.cached_industrial += 1
            elif result_type == "failed":
                self.failed += 1
            elif result_type == "skipped_urban":
                self.skipped_urban += 1
            elif result_type == "skipped_no_sat":
                self.skipped_no_sat += 1
            elif result_type == "skipped_black":
                self.skipped_black += 1
                
            if self.target_images > 0 and (self.downloaded + self.downloaded_industrial + self.cached + self.cached_industrial) >= self.target_images:
                self.stop_requested = True

def append_to_csv_sync(filepath, record, columns):
    df_new = pd.DataFrame([record], columns=columns)
    with csv_lock:
        if os.path.exists(filepath):
            try:
                df_existing = pd.read_csv(filepath)
                if 'event_id' in df_existing.columns and record.get('event_id') in df_existing['event_id'].values:
                    df_existing = df_existing[df_existing['event_id'] != record.get('event_id')]
                    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                    df_combined.to_csv(filepath, index=False)
                else:
                    df_new.to_csv(filepath, mode='a', header=False, index=False)
            except Exception as e:
                logger.warning(f"Error reading {filepath} for deduplication, appending anyway: {e}")
                df_new.to_csv(filepath, mode='a', header=False, index=False)
        else:
            df_new.to_csv(filepath, index=False)

def update_state(state_dict, state_file, event_id, status):
    with csv_lock:
        state_dict[event_id] = status
        tmp_path = f"{state_file}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            with open(tmp_path, 'w') as f:
                json.dump(state_dict, f)
            
            max_retries = 20
            for i in range(max_retries):
                try:
                    os.replace(tmp_path, state_file)
                    break
                except PermissionError as e:
                    if i == max_retries - 1:
                        logger.error(f"Failed to write state for {event_id} after {max_retries} retries: {e}")
                        raise
                    time.sleep(0.05)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

def process_event(event, args, paths, schemas, tracker, state_dict, state_file, dashboard):
    if tracker.stop_requested:
        return "skipped_stopped"
        
    event_id = event['event_id']
    lat = event['latitude']
    lon = event['longitude']
    date_str = event['date']
    time_str = event['time']
    source_file = event.get('source_file', 'unknown')
    
    update_state(state_dict, state_file, event_id, "PENDING")
    dashboard.update(current_event=event_id, status="Generating satellite images...", process="Fetching FIRMS data...")
    
    event_record = {
        "event_id": event_id,
        "source": source_file,
        "latitude": lat,
        "longitude": lon,
        "fire_event_date": date_str,
        "fire_event_time": time_str
    }
    append_to_csv_sync(paths['events'], event_record, schemas['events'])
    
    # Offline Prefilter
    urban_check = is_in_urban_area(lat, lon)
    if urban_check == "URBAN_FILTER_VALIDATION_FAILED":
        update_state(state_dict, state_file, event_id, "FAILED")
        tracker.add_result("skipped_urban")
        dashboard.update(process="Processing next event...")
        return "URBAN_FILTER_VALIDATION_FAILED"
    elif not urban_check:
        update_state(state_dict, state_file, event_id, "FAILED")
        tracker.add_result("skipped_urban")
        dashboard.update(process="Processing next event...")
        return "skipped_urban"
    
    # Offline Industrial Filter
    is_industrial = False
    industrial_check = is_near_industrial(lat, lon)
    if industrial_check == "INDUSTRIAL_FILTER_VALIDATION_FAILED":
        update_state(state_dict, state_file, event_id, "FAILED")
        tracker.add_result("skipped_industrial") # Keep this just for failure tracking
        dashboard.update(process="Processing next event...")
        return "INDUSTRIAL_FILTER_VALIDATION_FAILED"
    elif industrial_check:
        is_industrial = True
    
    # Generate geographic evidence record using the offline Natural Earth data
    geo_record = {
        "event_id": event_id,
        "latitude": lat,
        "longitude": lon,
        "landcover_source": "Natural Earth",
        "landcover_version": "10m_urban_areas",
        "landcover_resolution_m": 10,
        "source_url": "https://www.naturalearthdata.com/downloads/10m-cultural-vectors/10m-urban-area/"
    }
    append_to_csv_sync(paths['geographic'], geo_record, schemas['geographic'])
        
    update_state(state_dict, state_file, event_id, "GRID_ASSIGNED")
    dashboard.update(process="Checking event...")
    
    if tracker.stop_requested:
        return "skipped_stopped"
        
    # Satellite Search (Grid indexed automatically in search_satellite_imagery)
    item = search_satellite_imagery(
        lat=lat, 
        lon=lon, 
        date_str=date_str, 
        time_str=time_str,
        search_days=args.search_days,
        max_cloud=args.max_cloud
    )
    
    if not item:
        update_state(state_dict, state_file, event_id, "FAILED")
        tracker.add_result("skipped_no_sat")
        dashboard.update(process="Processing next event...")
        return "skipped_no_sat"
        
    update_state(state_dict, state_file, event_id, "SCENE_FOUND")
    dashboard.update(process="Searching satellite imagery...")
    
    sat_dt = item.datetime
    sat_date_str = sat_dt.strftime("%Y-%m-%d")
    sat_time_str = sat_dt.strftime("%H:%M:%S")
    
    event_dir = os.path.join(paths['events_dir'], event_id)
    
    if tracker.stop_requested:
        return "skipped_stopped"
        
    # Check if event_dir exists and has contents (basic cache check)
    dashboard.update(process="Downloading satellite imagery...")
    if os.path.exists(event_dir) and os.path.exists(os.path.join(event_dir, "metadata.json")):
        result_type = "cached"
        with open(os.path.join(event_dir, "metadata.json"), 'r') as f:
            download_result = json.load(f)
    else:
        download_result = download_and_crop_image(
            item=item,
            lat=lat,
            lon=lon,
            event_id=event_id,
            out_dir=event_dir,
            crop_km=args.crop_km,
            output_size=args.output_size,
            event_meta=event,
            mode=args.mode,
            is_industrial=is_industrial
        )
        
        if download_result and isinstance(download_result, dict) and "error" in download_result:
            update_state(state_dict, state_file, event_id, "FAILED")
            tracker.add_result("skipped_black")
            dashboard.update(process=f"SKIPPED: {download_result.get('error')}")
            return "skipped_black"
            
        result_type = "downloaded" if download_result else "failed"
    
    if result_type == "failed":
        # Transient network failure, don't lock as FAILED
        update_state(state_dict, state_file, event_id, "PENDING")
        tracker.add_result("failed")
        dashboard.update(process="Download error. Retrying later...")
        return "failed"
        
    if download_result:
        dashboard.update(process="Generating RGB...")
        update_state(state_dict, state_file, event_id, "IMAGE_DOWNLOADED")
        image_record = {
            "event_id": event_id,
            "satellite": "Sentinel-2",
            "satellite_acquisition_date": sat_date_str,
            "satellite_acquisition_time": sat_time_str,
            "event_dir": event_dir,
            "native_resolution": download_result.get("native_resolution", ""),
            "bands_available": download_result.get("bands_available", ""),
            "false_color_path": download_result.get("false_color_path", ""),
            "generation_timestamp": download_result.get("generation_timestamp", "")
        }
        append_to_csv_sync(paths['image_meta'], image_record, schemas['image_meta'])
        
        verif_record = {
            "event_id": event_id,
            "verification_status": "UNREVIEWED",
            "dataset_category": ""
        }
        append_to_csv_sync(paths['verification'], verif_record, schemas['verification'])
        
        update_state(state_dict, state_file, event_id, "COMPLETED")
        
        # Trigger building pipeline
        try:
            dashboard.update(process="Running building pipeline...")
            building_main = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "building-info-pipeline", "main.py"))
            subprocess.run([
                sys.executable, building_main,
                "--lat", str(lat),
                "--lon", str(lon),
                "--event-id", str(event_id),
                "--event-dir", str(event_dir)
            ], check=True, capture_output=True)
        except Exception as e:
            logger.error(f"Building pipeline failed for {event_id}: {e}")
            
        # Copy building image to YOLO dataset if the event was selected for YOLO
        import shutil
        building_png = os.path.join(event_dir, "institution_measurement.png")
        if os.path.exists(building_png):
            for split in ["train", "val", "test"]:
                yolo_img = os.path.join(os.path.dirname(__file__), "YOLO_dataset", "images", split, f"{event_id}.jpg")
                if os.path.exists(yolo_img):
                    yolo_bldg_dir = os.path.join(os.path.dirname(__file__), "YOLO_dataset", "building_images", split)
                    os.makedirs(yolo_bldg_dir, exist_ok=True)
                    shutil.copy2(building_png, os.path.join(yolo_bldg_dir, f"{event_id}.png"))
                    break
        
        if is_industrial:
            if result_type == "downloaded":
                result_type = "downloaded_industrial"
            elif result_type == "cached":
                result_type = "cached_industrial"
            
        tracker.add_result(result_type)
        dashboard.update(process="Image generated.", images_generated=tracker.downloaded + tracker.downloaded_industrial + tracker.cached + tracker.cached_industrial)
        return result_type
    else:
        return "failed"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", help="Run interactive CLI prompt")
    parser.add_argument("--source", type=str, default="VIIRS")
    parser.add_argument("--loc-type", type=str, default="World")
    parser.add_argument("--loc-val", type=str, default="World")
    parser.add_argument("--bbox", type=str, default=None, help="comma-separated min_lat,max_lat,min_lon,max_lon")
    parser.add_argument("--sat", type=str, default="Sentinel-2")
    parser.add_argument("--mode", type=str, default="BASELINE_B4_B11_B12", choices=["BASELINE_B4_B11_B12", "B8A_AUXILIARY"])
    parser.add_argument("--start-date", type=str, default="01-01-2025")
    parser.add_argument("--end-date", type=str, default="28-02-2025")
    
    parser.add_argument("--csv-dir", type=str, default="data/csv")
    parser.add_argument("--dataset-dir", type=str, default="dataset")
    parser.add_argument("--search-days", type=int, default=3)
    parser.add_argument("--max-cloud", type=float, default=30.0)
    parser.add_argument("--crop-km", type=float, default=2.0)
    parser.add_argument("--output-size", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--target-images", type=int, default=10)
    parser.add_argument("--urban-threshold", type=float, default=20.0)
    parser.add_argument("--max-workers", type=int, default=20)
    
    args, _ = parser.parse_known_args()
    
    if args.interactive or len(sys.argv) == 1:
        user_settings = run_cli()
        args.target_images = user_settings["target_images"]
        args.mode = user_settings["mode"]
    else:
        bbox_tuple = None
        if args.bbox:
            parts = [float(x.strip()) for x in args.bbox.split(",")]
            if len(parts) == 4:
                bbox_tuple = (parts[0], parts[1], parts[2], parts[3])
        
        user_settings = {
            "source": args.source,
            "loc_type": args.loc_type,
            "loc_val": args.loc_val,
            "bbox": bbox_tuple,
            "sat": args.sat,
            "start_str": args.start_date,
            "end_str": args.end_date,
            "start_dt": datetime.strptime(args.start_date, "%d-%m-%Y"),
            "end_dt": datetime.strptime(args.end_date, "%d-%m-%Y"),
            "target_images": args.target_images
        }
    
    init_cache()
    init_offline_filter()
    init_industrial_filter()
    
    events_dir = os.path.join(args.dataset_dir, "unreviewed", "events")
    metadata_dir = os.path.join(args.dataset_dir, "metadata")
    state_file = os.path.join(metadata_dir, "pipeline_state.json")
    
    paths = {
        'events_dir': events_dir,
        'events': os.path.join(metadata_dir, "events.csv"),
        'geographic': os.path.join(metadata_dir, "geographic_evidence.csv"),
        'image_meta': os.path.join(metadata_dir, "image_metadata.csv"),
        'verification': os.path.join(metadata_dir, "verification.csv")
    }
    
    for d in [events_dir, metadata_dir]:
        os.makedirs(d, exist_ok=True)
        
    state_dict = {}
    if os.path.exists(state_file):
        with open(state_file, 'r') as f:
            state_dict = json.load(f)
    
    start_time = time.time()
    
    dashboard = Dashboard(args.target_images)
    dashboard.update(
        status="Fetching FIRMS data...", 
        process="Fetching fire data...",
        data_source=user_settings['source'],
        location_str=user_settings['loc_val'],
        date_str=f"{user_settings['start_str']} -> {user_settings['end_str']}"
    )
    
    # Internal Verification Log (optional debug output)
    logger.info(f"Requested FIRMS period:\n{user_settings['start_str']} -> {user_settings['end_str']}")
    
    # API key check
    api_key = get_firms_api_key()
    if not api_key:
        print("\nUnable to retrieve FIRMS data.\n\nPlease check the data source and try again.\n")
        return
    
    # Existing FIRMS pipeline handles its own requests, errors, and parsing
    # Defer bounding box resolution to existing pipeline logic
    bbox = user_settings['bbox']
    if user_settings['loc_type'] == 'Country':
        dashboard.update(status="Geocoding Location...", process=f"Resolving bounding box for {user_settings['loc_val']}...")
        dashboard.render()
        bbox = get_bounding_box(user_settings['loc_val'])
        if not bbox:
            dashboard.update(status="Failed", process="Could not resolve location.")
            dashboard.render()
            print(f"\nNo geographic boundary found for: {user_settings['loc_val']}\n")
            return
            
    dashboard.update(status="Fetching FIRMS data...", process="Downloading and parsing CSVs...")
    dashboard.render()
    
    events = load_events_from_csv(
        start_date_obj=user_settings['start_dt'],
        end_date_obj=user_settings['end_dt'],
        source_choice=user_settings['source'],
        bbox=bbox
    )
    
    if not events:
        dashboard.update(status="Failed", process="No fire events found for the selected\ndate/location.")
        dashboard.render()
        print("\nNo fire events found for the selected date/location.\n")
        return
        
    random.seed(42)
    random.shuffle(events)
    if args.limit > 0:
        events = events[:args.limit]
        
    schemas = {
        'events': ["event_id", "source", "latitude", "longitude", "fire_event_date", "fire_event_time"],
        'geographic': [
            "event_id", "latitude", "longitude", "landcover_source", "landcover_version", 
            "landcover_resolution_m", "source_url"
        ],
        'image_meta': [
            "event_id", "satellite", "satellite_acquisition_date", "satellite_acquisition_time", 
            "event_dir", "native_resolution", "bands_available", "false_color_path", "generation_timestamp"
        ],
        'verification': ["event_id", "verification_status", "dataset_category"]
    }
    
    events_to_process = [e for e in events if state_dict.get(e['event_id']) not in ["COMPLETED", "FAILED"]]
    
    tracker = ProgressTracker(len(events_to_process), args.target_images)
    
    def monitor_progress():
        print("\nStarting generation... (Press Ctrl+C to abort)")
        while not tracker.stop_requested and tracker.processed < tracker.total:
            dashboard.update(
                images_generated=f"{tracker.downloaded + tracker.downloaded_industrial + tracker.cached + tracker.cached_industrial}",
                process=f"Processed: {tracker.processed}/{tracker.total} | Failed: {tracker.failed} | Rejected: {tracker.skipped_urban + tracker.skipped_no_sat + tracker.skipped_black}"
            )
            dashboard.render()
            time.sleep(0.1)
            
    monitor_thread = threading.Thread(target=monitor_progress, daemon=True)
    monitor_thread.start()
    
    import concurrent.futures
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        active_futures = set()
        event_iter = iter(events_to_process)
        
        while not tracker.stop_requested:
            successful = tracker.downloaded + tracker.downloaded_industrial + tracker.cached + tracker.cached_industrial
            remaining = args.target_images - successful
            
            if remaining <= 0:
                tracker.stop_requested = True
                break
                
            allowed_active = min(remaining, args.max_workers)
            
            while len(active_futures) < allowed_active:
                try:
                    event = next(event_iter)
                    # TARGET_IMAGES strictly counts successfully generated imagery.
                    # Submitting a job does not count towards the target.
                    future = executor.submit(process_event, event, args, paths, schemas, tracker, state_dict, state_file, dashboard)
                    active_futures.add(future)
                except StopIteration:
                    break
                    
            if not active_futures:
                break
                
            done, active_futures = concurrent.futures.wait(
                active_futures,
                return_when=concurrent.futures.FIRST_COMPLETED
            )
            
            for future in done:
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Worker exception: {e}")
                    
        # When target is reached, cancel pending tasks and exit immediately.
        executor.shutdown(wait=False, cancel_futures=True)
        
    tracker.stop_requested = True 
    dashboard.status = "Completed"
    dashboard.process = "Pipeline completed successfully."
    dashboard.render()
    print("\n============================================================")
    print("Pipeline completed successfully.")
    print(f"Images generated: {tracker.downloaded + tracker.cached} (True) + {tracker.downloaded_industrial + tracker.cached_industrial} (Industrial) = {tracker.downloaded + tracker.cached + tracker.downloaded_industrial + tracker.cached_industrial} Total")
    print(f"Failed events: {tracker.failed}")
    print(f"Rejected events: {tracker.skipped_urban + tracker.skipped_no_sat + tracker.skipped_black}")
    print(f"Output directory: {paths['events_dir']}")
    print("============================================================")
    logger.info(f"Pipeline completed in {time.time() - start_time:.2f} seconds.")
    
    # Force exit to prevent hanging on active threads (safe due to csv_lock)
    with csv_lock:
        os._exit(0)

if __name__ == "__main__":
    main()

