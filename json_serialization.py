"""
JSON Serialization Helper

This module provides utilities to ensure proper JSON serialization
for all data types used in the trading bot.
"""

import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, time
from decimal import Decimal
from typing import Any, Dict, List, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class JSONEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles special data types:
    - datetime, date, time objects
    - Decimal objects
    - numpy numeric types
    - pandas NA/NaN values
    - boolean values
    - sets and other iterables
    """
    
    def default(self, obj: Any) -> Any:
        # Handle datetime objects
        if isinstance(obj, (datetime, date, time)):
            return obj.isoformat()
            
        # Handle Decimal objects
        if isinstance(obj, Decimal):
            return float(obj)
            
        # Handle numpy numeric types
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
            return int(obj)
            
        if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
            
        if isinstance(obj, np.bool_):
            return bool(obj)
            
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()
            
        # Handle pandas NA/NaN
        if pd.isna(obj):
            return None
            
        # Handle sets and other iterables
        if isinstance(obj, set):
            return list(obj)
            
        # Let the base class handle it or raise TypeError
        return super().default(obj)

def json_serialize(obj: Any) -> str:
    """
    Serialize an object to JSON string with proper handling of special data types.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string
    """
    try:
        return json.dumps(obj, cls=JSONEncoder)
    except Exception as e:
        logger.error(f"Error serializing to JSON: {str(e)}")
        
        # Try to sanitize the object
        sanitized_obj = _sanitize_for_json(obj)
        return json.dumps(sanitized_obj, cls=JSONEncoder)

def json_deserialize(json_str: str) -> Any:
    """
    Deserialize a JSON string to an object.
    
    Args:
        json_str: JSON string
        
    Returns:
        Deserialized object
    """
    try:
        return json.loads(json_str)
    except Exception as e:
        logger.error(f"Error deserializing from JSON: {str(e)}")
        return None

def _sanitize_for_json(obj: Any) -> Any:
    """
    Recursively sanitize an object to ensure it can be serialized to JSON.
    
    Args:
        obj: Object to sanitize
        
    Returns:
        Sanitized object
    """
    if obj is None:
        return None
        
    if isinstance(obj, (str, int, float, bool)):
        return obj
        
    if isinstance(obj, (datetime, date, time)):
        return obj.isoformat()
        
    if isinstance(obj, Decimal):
        return float(obj)
        
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
        
    if isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
        
    if isinstance(obj, np.bool_):
        return bool(obj)
        
    if isinstance(obj, np.ndarray):
        return _sanitize_for_json(obj.tolist())
        
    if pd.isna(obj):
        return None
        
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
        
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in obj]
        
    # For other types, convert to string
    try:
        return str(obj)
    except:
        return "UNSERIALIZABLE_OBJECT"

def safe_json_dumps(obj: Any, default: str = None) -> str:
    """
    Safely convert an object to a JSON string, with fallback to default.
    
    Args:
        obj: Object to convert
        default: Default value if conversion fails
        
    Returns:
        JSON string or default
    """
    try:
        return json_serialize(obj)
    except:
        return default if default is not None else "{}"

def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """
    Safely parse a JSON string, with fallback to default.
    
    Args:
        json_str: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed object or default
    """
    try:
        return json_deserialize(json_str)
    except:
        return default if default is not None else {}
