import os
import pandas as pd
import logging
from datetime import timedelta
from config.settings import FIRMS_API_BASE, CSV_DIR

logger = logging.getLogger(__name__)

def get_firms_api_key():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env_paths = [
        os.path.join(base_dir, ".env"),
        os.path.join(os.path.dirname(base_dir), ".env")
    ]
    for env_path in env_paths:
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith("NASA_FIRMS_API_KEY="):
                        return line.strip().split("=")[1]
    return None

def plan_required_requests(start_date_obj, end_date_obj, source_choice):
    """
    Generates a list of required API sources and dates.
    source_choice comes from CLI: "VIIRS", "MODIS", "VIIRS + MODIS"
    """
    api_sources = []
    if "VIIRS" in source_choice:
        api_sources.append("VIIRS_SNPP")
    if "MODIS" in source_choice:
        api_sources.append("MODIS")
        
    dates = []
    current_date = start_date_obj
    while current_date <= end_date_obj:
        dates.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
        
    requests = []
    for date_str in dates:
        for source in api_sources:
            requests.append({"source": source, "date": date_str})
            
    return requests

def _download_single_firms_req(req, api_key, session):
    date_str = req['date']
    source = req['source']
    
    output_filename = f"firms_{source}_{date_str}.csv"
    output_path = os.path.join(CSV_DIR, output_filename)
    
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return
        
    logger.info(f"Downloading NASA FIRMS data for {source} on {date_str}...")
    
    url_sp = f"{FIRMS_API_BASE}/{api_key}/{source}_SP/world/1/{date_str}"
    tmp_path = output_path + ".tmp"
    
    try:
        response = session.get(url_sp, timeout=30)
        response.raise_for_status()
        content = response.text
        
        if len(content.strip().split('\n')) <= 1:
            url_nrt = f"{FIRMS_API_BASE}/{api_key}/{source}_NRT/world/1/{date_str}"
            response = session.get(url_nrt, timeout=30)
            response.raise_for_status()
            content = response.text
            
        with open(tmp_path, 'w') as f:
            f.write(content)
            
        os.rename(tmp_path, output_path)
        logger.info(f"Saved FIRMS data to {output_path}")
    except Exception as e:
        logger.error(f"Failed to fetch FIRMS data for {date_str}: {e}")
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def fetch_firms_date_range(requests_list):
    """
    Downloads ONLY the required missing FIRMS data concurrently.
    Uses Session reuse and atomic tmp files.
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from config.settings import DEFAULT_MAX_WORKERS
    
    api_key = get_firms_api_key()
    if not api_key:
        logger.error("NASA_FIRMS_API_KEY not found in .env. Cannot download FIRMS data.")
        return
        
    os.makedirs(CSV_DIR, exist_ok=True)
    
    # Use a session with a connection pool sized to max workers and exponential backoff
    session = requests.Session()
    from urllib3.util import Retry
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=DEFAULT_MAX_WORKERS, 
        pool_maxsize=DEFAULT_MAX_WORKERS,
        max_retries=retries
    )
    session.mount('https://', adapter)
    
    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as executor:
        futures = []
        for req in requests_list:
            futures.append(executor.submit(_download_single_firms_req, req, api_key, session))
            
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"Worker error in FIRMS download: {e}")
                
    session.close()

def load_events_from_csv(start_date_obj, end_date_obj, source_choice, bbox=None):
    """
    Reads ONLY the required CSV files from CSV_DIR, standardizes them,
    and returns a list of fire event dictionaries.
    Uses vectorized pandas operations for performance.
    """
    requests_list = plan_required_requests(start_date_obj, end_date_obj, source_choice)
    fetch_firms_date_range(requests_list)
    
    all_dfs = []
    
    for req in requests_list:
        date_str = req['date']
        source = req['source']
        file_path = os.path.join(CSV_DIR, f"firms_{source}_{date_str}.csv")
        
        if not os.path.exists(file_path):
            continue
            
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                continue
                
            lat_col = next((col for col in df.columns if col.lower() in ['latitude', 'lat']), None)
            lon_col = next((col for col in df.columns if col.lower() in ['longitude', 'lon']), None)
            date_col = next((col for col in df.columns if col.lower() in ['acq_date', 'date']), None)
            
            if not lat_col or not lon_col or not date_col:
                continue
                
            # Rename columns to standard names
            df = df.rename(columns={lat_col: 'latitude', lon_col: 'longitude', date_col: 'date'})
            
            # Ensure acq_time exists
            if 'acq_time' not in df.columns:
                df['acq_time'] = ''
                
            # Basic validation
            df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
            df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
            df = df.dropna(subset=['latitude', 'longitude'])
            df = df[(df['latitude'] >= -90) & (df['latitude'] <= 90)]
            df = df[(df['longitude'] >= -180) & (df['longitude'] <= 180)]
            
            # Geographic filter if bbox provided
            if bbox:
                min_lat, max_lat, min_lon, max_lon = bbox
                df = df[(df['latitude'] >= min_lat) & (df['latitude'] <= max_lat)]
                df = df[(df['longitude'] >= min_lon) & (df['longitude'] <= max_lon)]
            
            if df.empty:
                continue
                
            # Date filtering
            df['date_obj'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
            df = df.dropna(subset=['date_obj'])
            
            if df.empty:
                continue
                
            # Convert start_date_obj and end_date_obj to naive datetime64[ns] equivalent comparison
            start_dt = pd.Timestamp(start_date_obj.replace(tzinfo=None))
            end_dt = pd.Timestamp(end_date_obj.replace(tzinfo=None))
            
            df = df[(df['date_obj'] >= start_dt) & (df['date_obj'] <= end_dt)]
            
            if df.empty:
                continue
                
            # Time string formatting
            def format_time(t):
                if pd.isna(t) or t == '': return ''
                try:
                    return str(int(float(t))).zfill(4)
                except Exception as e:
                    logger.debug(f"Failed to parse time {t}: {e}")
                    return ''
                    
            df['time'] = df['acq_time'].apply(format_time)
            
            df['source_file'] = os.path.basename(file_path)
            
            if 'satellite' not in df.columns:
                df['satellite'] = ''
                
            all_dfs.append(df[['latitude', 'longitude', 'date', 'time', 'satellite', 'source_file']])
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")

    if not all_dfs:
        return []
        
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Deduplication
    unique_df = combined_df.drop_duplicates(subset=['latitude', 'longitude', 'date', 'time', 'satellite'])
    
    events = []
    for idx, row in unique_df.iterrows():
        events.append({
            'event_id': f"event_{idx+1:06d}",
            'latitude': row['latitude'],
            'longitude': row['longitude'],
            'date': str(row['date']),
            'time': str(row['time']),
            'satellite': str(row['satellite']),
            'source_file': str(row['source_file'])
        })
        
    logger.info(f"Loaded {len(events)} unique events from {len(requests_list)} required periods.")
    return events
