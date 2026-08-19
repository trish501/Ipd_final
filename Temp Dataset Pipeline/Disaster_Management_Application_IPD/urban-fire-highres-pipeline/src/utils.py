import hashlib
from datetime import datetime, timezone

def get_hash_key(data: str) -> str:
    """Creates a consistent MD5 hash for cache keys."""
    return hashlib.md5(data.encode('utf-8')).hexdigest()

def parse_date_string(date_str: str) -> datetime:
    """
    Parses 'DD Month YYYY' or 'YYYY-MM-DD' into a UTC datetime object.
    Raises ValueError if format is invalid.
    """
    try:
        # Example: '01 January 2025'
        return datetime.strptime(date_str.strip(), "%d %B %Y").replace(tzinfo=timezone.utc)
    except ValueError:
        pass
    
    try:
        # Example: '2025-01-01'
        return datetime.strptime(date_str.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Could not parse date string: {date_str}")

def parse_datetime_string(date_str: str, time_str: str = "") -> datetime:
    """
    Parses a date and time combination into a UTC datetime object.
    """
    if not time_str:
        return parse_date_string(date_str)
        
    dt_str = f"{date_str.strip()} {time_str.strip()}"
    try:
        # Expected FIRMS ISO date or equivalent 
        if ":" in time_str:
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        else:
            return datetime.strptime(dt_str, "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
    except ValueError:
        raise ValueError(f"Could not parse datetime string: {dt_str}")
