import os
from datetime import datetime
from src.fire_data import load_events_from_csv
from src.offline_urban_filter import init_offline_filter
from src.cache import init_cache

import main
def monkey_update_state(state_dict, state_file, event_id, status):
    state_dict[event_id] = status
    # skip writing to file for the batch to avoid PermissionError
main.update_state = monkey_update_state

from main import process_event, Dashboard, ProgressTracker

def run_batch():
    init_cache()
    init_offline_filter()
    
    start_dt = datetime.strptime("01-08-2021", "%d-%m-%Y")
    end_dt = datetime.strptime("31-08-2021", "%d-%m-%Y")
    
    # Northern California Bounding Box
    bbox = (39.0, 42.0, -124.0, -120.0)
    
    print("Loading FIRMS events...")
    events = load_events_from_csv(
        start_date_obj=start_dt,
        end_date_obj=end_dt,
        source_choice="VIIRS",
        bbox=bbox
    )
    print(f"Loaded {len(events)} events.")
    
    if not events:
        print("No events found!")
        return
        
    class MockArgs:
        search_days = 3
        max_cloud = 30.0
        crop_km = 2.0
        output_size = 1024
        
    args = MockArgs()
    
    paths = {
        'events_dir': "dataset/unreviewed/events",
        'events': "dataset/metadata/events.csv",
        'geographic': "dataset/metadata/geographic_evidence.csv",
        'image_meta': "dataset/metadata/image_metadata.csv",
        'verification': "dataset/metadata/verification.csv"
    }
    
    schemas = {
        'events': ["event_id", "source", "latitude", "longitude", "fire_event_date", "fire_event_time"],
        'geographic': ["event_id", "latitude", "longitude", "landcover_source", "landcover_version", "landcover_resolution_m", "source_url"],
        'image_meta': ["event_id", "satellite", "satellite_acquisition_date", "satellite_acquisition_time", "event_dir", "native_resolution", "bands_available", "rgb_path", "swir_path", "swir_nir_path", "generation_timestamp"],
        'verification': ["event_id", "verification_status", "dataset_category"]
    }
    
    os.makedirs(paths['events_dir'], exist_ok=True)
    os.makedirs("dataset/metadata", exist_ok=True)
    
    target_count = 15
    tracker = ProgressTracker(len(events), target_count)
    dashboard = Dashboard(target_count)
    state_dict = {}
    
    processed_count = 0
    successful_count = 0
    
    for event in events:
        processed_count += 1
        res = process_event(event, args, paths, schemas, tracker, state_dict, "dataset/metadata/pipeline_state.json", dashboard)
        if res not in ["skipped_urban", "skipped_no_sat", "skipped_black"]:
            successful_count += 1
            print(f"Event {processed_count}: {event['event_id']} processed successfully (Result: {res}). Total successful: {successful_count}")
        if successful_count >= target_count:
            break
            
    print(f"Done processing. Successfully completed {successful_count} events after checking {processed_count} events.")

if __name__ == "__main__":
    run_batch()
