import argparse
import logging
import json
import sys
import os
import csv
from dataclasses import asdict

from dimensions.models import SIHInput
from dimensions.pipeline import DimensionsPipeline

def print_separator():
    print("==================================================")

def print_thin_separator():
    print("--------------------------------------------------")

def main_menu():
    while True:
        print_separator()
        print("       SIH INSTITUTION DIMENSIONS SYSTEM")
        print_separator()
        print("\nFind and measure institution buildings using")
        print("high-resolution RGB satellite imagery.\n")
        print_thin_separator()
        print("\nChoose input mode:")
        print("1. Single location (Institution)")
        print("2. Fire event (Event ID, Lat, Lon)")
        print("3. Batch CSV (Process multiple events)")
        print("4. Exit\n")
        
        choice = input("Enter your choice: ").strip()
        if choice in ['1', '2', '3', '4']:
            return choice
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")

def validate_coords(lat_str, lon_str):
    try:
        lat = float(lat_str)
        if not (-90 <= lat <= 90):
            print("\nError: Latitude must be between -90 and 90.")
            return None, None
    except ValueError:
        print("\nError: Latitude must be a number.")
        return None, None
        
    try:
        lon = float(lon_str)
        if not (-180 <= lon <= 180):
            print("\nError: Longitude must be between -180 and 180.")
            return None, None
    except ValueError:
        print("\nError: Longitude must be a number.")
        return None, None
        
    return lat, lon

def collect_institution_inputs():
    while True:
        print_separator()
        print("                 INPUT FORM")
        print_separator()
        
        institution_name = input("\nInstitution / College Name:\n").strip()
        if not institution_name:
            print("\nError: Institution name is required.")
            continue
            
        city = input("\nCity:\n").strip()
        
        lat_str = input("\nLatitude:\n").strip()
        lon_str = input("\nLongitude:\n").strip()
        
        lat, lon = validate_coords(lat_str, lon_str)
        if lat is None or lon is None:
            input("Press Enter to try again...")
            continue
            
        return SIHInput(latitude=lat, longitude=lon, institution_name=institution_name, city=city)

def collect_fire_event_inputs():
    while True:
        print_separator()
        print("             FIRE EVENT INPUT")
        print_separator()
        
        event_id = input("\nEvent ID:\n").strip()
        if not event_id:
            print("\nError: Event ID is required.")
            continue
            
        lat_str = input("\nLatitude:\n").strip()
        lon_str = input("\nLongitude:\n").strip()
        
        lat, lon = validate_coords(lat_str, lon_str)
        if lat is None or lon is None:
            input("Press Enter to try again...")
            continue
            
        return SIHInput(latitude=lat, longitude=lon, event_id=event_id)

def collect_batch_inputs():
    while True:
        print_separator()
        print("             BATCH CSV INPUT")
        print_separator()
        
        csv_path = input("\nPath to CSV file:\n").strip()
        if not os.path.exists(csv_path):
            print(f"\nError: File '{csv_path}' not found.")
            input("Press Enter to try again...")
            continue
            
        return csv_path

def confirm_inputs(input_data):
    while True:
        print_separator()
        print("INPUT CONFIRMATION")
        print_separator()
        
        if getattr(input_data, 'event_id', None):
            print(f"\nEvent ID:\n{input_data.event_id}")
        if getattr(input_data, 'institution_name', None):
            print(f"\nInstitution:\n{input_data.institution_name}")
            print(f"\nCity:\n{input_data.city}")
            
        print(f"\nLatitude:\n{input_data.latitude:.8f}")
        print(f"\nLongitude:\n{input_data.longitude:.8f}\n")
        
        print_separator()
        print("\n1. Start")
        print("2. Edit")
        print("3. Cancel\n")
        
        choice = input("Enter choice: ").strip()
        if choice in ['1', '2', '3']:
            return choice
        else:
            print("\nInvalid choice. Please enter 1, 2, or 3.")

def execute_pipeline(input_data):
    print_separator()
    print("PROCESSING")
    print_separator()
    if input_data.event_id:
        print(f"\nEvent ID:\n{input_data.event_id}")
    else:
        print(f"\nInstitution:\n{input_data.institution_name}")
    print(f"\nLocation:\n{input_data.latitude:.8f}, {input_data.longitude:.8f}\n")
    print_thin_separator()
    print("\n[1/6] Validating location...")
    print("[2/6] Finding building candidates...")
    print("[3/6] Selecting target area...")
    print("[4/6] Retrieving high-resolution RGB imagery...")
    print("[5/6] Calculating building dimensions...")
    print("[6/6] Rendering final PNG...\n")
    print("Please wait...\n")
    print_separator()
    
    pipeline = DimensionsPipeline()
    result = pipeline.run(input_data)
    
    if result.status == "SUCCESS":
        display_success(result)
    else:
        display_error(result)

def display_success(result):
    print_separator()
    print("        MEASUREMENT COMPLETE")
    print_separator()
    
    if result.input.event_id:
        print(f"\nEvent ID:\n{result.input.event_id}")
    else:
        print(f"\nInstitution:\n{result.input.institution_name}")
    print(f"\nLocation:\n{result.input.latitude:.8f}, {result.input.longitude:.8f}\n")
    print_thin_separator()
    
    print(f"\nSelected Buildings:\n{result.institution.selected_buildings}")
    print(f"\nTotal Building Area:\n{result.institution.total_building_footprint_area_sq_m:,.2f} m²")
    print(f"\nTotal Building Area:\n{result.institution.total_building_footprint_area_sq_ft:,.2f} ft²")
    
    print(f"\nImagery:\n{result.imagery.provider}")
    print(f"\nNative Resolution:\n~{result.imagery.native_resolution_m:.1f} m")
    print("\nFinal Image:\nPNG\n")
    
    print_thin_separator()
    print("\nOUTPUT\n")
    
    if result.input.event_dir:
        out_folder = result.input.event_dir
    elif result.input.event_id:
        out_folder = f"dimensions/outputs/{result.input.event_id}"
    else:
        inst_name = result.input.institution_name.replace(' ', '_') if result.input.institution_name else "Fire_Event"
        out_folder = f"dimensions/outputs/{inst_name}_{result.input.latitude}_{result.input.longitude}"
    
    print(f"Image:\n{out_folder}/institution_measurement.png")
    print(f"Measurements:\n{out_folder}/measurements.csv")
    print(f"Building Data:\n{out_folder}/selected_buildings.geojson")
    print(f"Summary:\n{out_folder}/summary.json\n")
    print_separator()

def display_error(result):
    print_separator()
    print("PROCESSING FAILED")
    print_separator()
    print(f"\nReason:\n{result.error or result.status}\n")
    print_thin_separator()
    print("\nPossible action:\nPlease check the coordinates or try a different location.")
    print_separator()

def run_interactive():
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("dimensions").setLevel(logging.WARNING)
    
    while True:
        mode = main_menu()
        if mode == '4':
            print("\nExiting...")
            break
            
        if mode == '3':
            csv_path = collect_batch_inputs()
            process_batch(csv_path)
            input("\nPress Enter to return to main menu...")
            continue
            
        while True:
            if mode == '1':
                input_data = collect_institution_inputs()
            elif mode == '2':
                input_data = collect_fire_event_inputs()
                
            confirm = confirm_inputs(input_data)
            if confirm == '1':
                execute_pipeline(input_data)
                
                print("\n1. Run another")
                print("2. Return to main menu")
                print("3. Exit\n")
                
                after_choice = input("Enter choice: ").strip()
                if after_choice == '1':
                    continue
                elif after_choice == '2':
                    break
                else:
                    print("\nExiting...")
                    return
            
            elif confirm == '2':
                continue
            elif confirm == '3':
                break

def process_batch(csv_path):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if not os.path.exists(csv_path):
        logger.error(f"Batch file not found: {csv_path}")
        return 1
        
    pipeline = DimensionsPipeline()
    success_count = 0
    fail_count = 0
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            event_id = row.get("event_id")
            lat_str = row.get("latitude")
            lon_str = row.get("longitude")
            
            if not event_id or not lat_str or not lon_str:
                logger.error(f"Skipping invalid row: {row}")
                fail_count += 1
                continue
                
            try:
                lat = float(lat_str)
                lon = float(lon_str)
            except ValueError:
                logger.error(f"Invalid coordinates for {event_id}: {lat_str}, {lon_str}")
                fail_count += 1
                continue
                
            logger.info(f"Processing batch event {event_id} at {lat}, {lon}")
            input_data = SIHInput(latitude=lat, longitude=lon, event_id=event_id)
            result = pipeline.run(input_data)
            
            if result.status == "SUCCESS":
                success_count += 1
            else:
                logger.error(f"Event {event_id} failed: {result.error or result.status}")
                fail_count += 1
                
    logger.info(f"Batch processing complete. Success: {success_count}, Failed: {fail_count}")
    return 0 if fail_count == 0 else 1

def run_non_interactive(args):
    if args.batch:
        return process_batch(args.batch)
        
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    if not args.lat or not args.lon:
        print("Error: --lat and --lon are required unless --batch is used.")
        return 1
        
    input_data = SIHInput(
        latitude=args.lat,
        longitude=args.lon,
        institution_name=args.institution_name,
        city=args.city,
        event_id=args.event_id,
        event_dir=args.event_dir
    )
    
    pipeline = DimensionsPipeline()
    try:
        result = pipeline.run(input_data)
    except Exception as e:
        logger.error(f"Unhandled pipeline exception: {e}")
        from dimensions.models import SIHResult, SIHImagerySummary, SIHInstitutionSummary
        result = SIHResult(
            status="FAILED",
            input=input_data,
            imagery=SIHImagerySummary(),
            institution=SIHInstitutionSummary(),
            buildings=[],
            error=str(e)
        )
        if input_data.event_dir:
            out_folder = input_data.event_dir
            os.makedirs(out_folder, exist_ok=True)
            pipeline.exporter.export_all(result, out_folder)
        elif input_data.event_id:
            import os
            from dimensions.config import OUTPUTS_DIR
            out_folder = os.path.join(OUTPUTS_DIR, input_data.event_id)
            os.makedirs(out_folder, exist_ok=True)
            pipeline.exporter.export_all(result, out_folder)
            
    res_dict = asdict(result)
    for b in res_dict.get("buildings", []):
        b.pop("geometry_wgs84", None)
        
    print(json.dumps(res_dict, indent=4))
    
    if result.status not in ["SUCCESS", "NO_BUILDINGS_FOUND"]:
        return 1
    return 0

def main():
    if len(sys.argv) == 1:
        run_interactive()
        return 0
        
    parser = argparse.ArgumentParser(description="SIH Institution & Fire Event Measurement")
    parser.add_argument("--lat", type=float, required=False)
    parser.add_argument("--lon", type=float, required=False)
    parser.add_argument("--institution-name", type=str, required=False)
    parser.add_argument("--city", type=str, required=False)
    parser.add_argument("--event-id", type=str, required=False)
    parser.add_argument("--batch", type=str, required=False, help="Path to CSV containing event_id, latitude, longitude")
    parser.add_argument("--event-dir", type=str, required=False, help="Path to event directory from Sentinel pipeline")
    
    args = parser.parse_args()
    
    if not args.batch:
        if not args.lat or not args.lon:
            parser.error("--lat and --lon are required when not using --batch")
        if not args.institution_name and not args.event_id:
            parser.error("Either --institution-name or --event-id is required")
            
    return run_non_interactive(args)

if __name__ == "__main__":
    sys.exit(main())
