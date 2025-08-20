"""Event logging system for experiments"""

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import threading


class EventLogger:
    """JSONL event logger for experiment tracking"""
    
    def __init__(self, log_file: str):
        self.log_file = Path(log_file)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.event_count = 0
        
    def log_event(self, event_type: str, data: Dict[str, Any], metadata: Optional[Dict] = None):
        """Log an event to JSONL file"""
        
        event = {
            "timestamp": time.time(),
            "session_id": self.session_id,
            "event_id": self.event_count,
            "event_type": event_type,
            "data": data
        }
        
        if metadata:
            event["metadata"] = metadata
            
        # Thread-safe write
        with self.lock:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
            self.event_count += 1
            
    def log_error(self, error: Exception, context: Dict[str, Any] = None):
        """Log an error event"""
        
        self.log_event("error", {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        })
        
    def log_metric(self, metric_name: str, value: Any, tags: Dict[str, str] = None):
        """Log a metric value"""
        
        self.log_event("metric", {
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {}
        })
        
    def read_events(self) -> list:
        """Read all events from log file"""
        
        events = []
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                for line in f:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return events
    
    def get_events_by_type(self, event_type: str) -> list:
        """Get all events of a specific type"""
        
        events = self.read_events()
        return [e for e in events if e.get("event_type") == event_type]
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logged metrics"""
        
        metric_events = self.get_events_by_type("metric")
        
        summary = {}
        for event in metric_events:
            metric_name = event["data"]["metric_name"]
            value = event["data"]["value"]
            
            if metric_name not in summary:
                summary[metric_name] = {
                    "values": [],
                    "count": 0,
                    "mean": 0,
                    "min": float('inf'),
                    "max": float('-inf')
                }
                
            summary[metric_name]["values"].append(value)
            summary[metric_name]["count"] += 1
            
        # Calculate statistics
        for metric_name, data in summary.items():
            values = data["values"]
            if values and all(isinstance(v, (int, float)) for v in values):
                data["mean"] = sum(values) / len(values)
                data["min"] = min(values)
                data["max"] = max(values)
            del data["values"]  # Remove raw values from summary
            
        return summary
