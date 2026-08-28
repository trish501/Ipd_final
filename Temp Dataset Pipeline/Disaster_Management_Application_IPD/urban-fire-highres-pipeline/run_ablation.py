import os
import json
import logging
from src.fire_data import get_firms_api_key
from src.satellite_search import search_satellite_imagery
from src.image_downloader import download_and_crop_image

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def run_ablation():
    events_file = "reviewed_events.json"
    if not os.path.exists(events_file):
        logger.error(f"Error: {events_file} not found. Please provide the list of reviewed events.")
        return

    with open(events_file, 'r') as f:
        events = json.load(f)

    api_key = get_firms_api_key()
    if not api_key:
        logger.error("Unable to retrieve FIRMS data. Please check your API key.")
        return

    report = {
        "total_events": len(events),
        "baseline_true_positives": 0,
        "baseline_false_positives": 0,
        "experimental_true_positives": 0,
        "experimental_false_positives": 0,
        "false_positive_suppression_rate": 0.0,
        "true_positives_lost": 0,
        "details": []
    }

    out_dir_base = "ablation_output"
    os.makedirs(out_dir_base, exist_ok=True)

    for event in events:
        event_id = event.get('event_id')
        lat = event.get('latitude')
        lon = event.get('longitude')
        date_str = event.get('date')
        time_str = event.get('time')
        ground_truth = event.get('ground_truth') # "TRUE_POSITIVE" or "FALSE_POSITIVE"

        logger.info(f"Processing event: {event_id}")

        item = search_satellite_imagery(
            lat=lat, 
            lon=lon, 
            date_str=date_str, 
            time_str=time_str,
            search_days=3,
            max_cloud=30.0
        )

        if not item:
            logger.warning(f"No imagery found for event {event_id}")
            continue
            
        event_report = {
            "event_id": event_id,
            "ground_truth": ground_truth,
            "baseline_detected": False,
            "experimental_detected": False
        }

        # 1. Run Baseline
        out_dir_baseline = os.path.join(out_dir_base, "baseline", event_id)
        result_baseline = download_and_crop_image(
            item=item,
            lat=lat,
            lon=lon,
            event_id=event_id,
            out_dir=out_dir_baseline,
            crop_km=2.0,
            output_size=1024,
            event_meta=event,
            mode="BASELINE_B4_B11_B12"
        )
        if result_baseline and "error" not in result_baseline:
            event_report["baseline_detected"] = True

        # 2. Run Experimental
        out_dir_exp = os.path.join(out_dir_base, "experimental", event_id)
        result_exp = download_and_crop_image(
            item=item,
            lat=lat,
            lon=lon,
            event_id=event_id,
            out_dir=out_dir_exp,
            crop_km=2.0,
            output_size=1024,
            event_meta=event,
            mode="EXPERIMENTAL_B8A_REVIEW"
        )
        if result_exp and "error" not in result_exp:
            event_report["experimental_detected"] = True

        report["details"].append(event_report)

        # Tally
        if ground_truth == "TRUE_POSITIVE":
            if event_report["baseline_detected"]:
                report["baseline_true_positives"] += 1
            if event_report["experimental_detected"]:
                report["experimental_true_positives"] += 1
        elif ground_truth == "FALSE_POSITIVE":
            if event_report["baseline_detected"]:
                report["baseline_false_positives"] += 1
            if event_report["experimental_detected"]:
                report["experimental_false_positives"] += 1

    # Compute metrics
    if report["baseline_false_positives"] > 0:
        suppressed = report["baseline_false_positives"] - report["experimental_false_positives"]
        report["false_positive_suppression_rate"] = suppressed / report["baseline_false_positives"]
    
    report["true_positives_lost"] = report["baseline_true_positives"] - report["experimental_true_positives"]

    report_file = os.path.join(out_dir_base, "ablation_report.json")
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=4)

    logger.info("========================================")
    logger.info("Ablation Run Completed")
    logger.info(f"Baseline TP: {report['baseline_true_positives']}, FP: {report['baseline_false_positives']}")
    logger.info(f"Experimental TP: {report['experimental_true_positives']}, FP: {report['experimental_false_positives']}")
    logger.info(f"FP Suppression Rate: {report['false_positive_suppression_rate']:.2%}")
    logger.info(f"TP Lost: {report['true_positives_lost']}")
    logger.info("========================================")

if __name__ == "__main__":
    run_ablation()
