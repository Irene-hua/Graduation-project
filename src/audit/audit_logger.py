"""
Audit Logger
Records system access, queries, and model invocations with integrity checking
"""

import logging
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import threading


class AuditLogger:
    """Audit logger with integrity verification"""
    
    def __init__(self, 
                 log_directory: str = './logs',
                 log_level: str = 'INFO',
                 enable_integrity_check: bool = True):
        """
        Initialize audit logger
        
        Args:
            log_directory: Directory for log files
            log_level: Logging level
            enable_integrity_check: Enable log integrity verification
        """
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(parents=True, exist_ok=True)
        
        self.enable_integrity_check = enable_integrity_check
        self.lock = threading.Lock()
        
        # Setup main logger
        self.logger = logging.getLogger('audit')
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create log file handler
        log_file = self.log_directory / f'audit_{datetime.now().strftime("%Y%m%d")}.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        
        # Integrity tracking
        self.last_hash = None
        if self.enable_integrity_check:
            self._load_last_hash()
        
        self.logger.info("Audit logger initialized")
    
    def _load_last_hash(self):
        """Load last hash from integrity file"""
        integrity_file = self.log_directory / 'integrity.txt'
        if integrity_file.exists():
            try:
                with open(integrity_file, 'r') as f:
                    self.last_hash = f.read().strip()
            except Exception as e:
                self.logger.warning(f"Could not load last hash: {e}")
    
    def _save_hash(self, hash_value: str):
        """Save hash to integrity file"""
        integrity_file = self.log_directory / 'integrity.txt'
        try:
            with open(integrity_file, 'w') as f:
                f.write(hash_value)
        except Exception as e:
            self.logger.error(f"Could not save hash: {e}")
    
    def _calculate_hash(self, data: str) -> str:
        """Calculate SHA256 hash with chain"""
        if self.last_hash:
            # Chain with previous hash
            combined = self.last_hash + data
        else:
            combined = data
        
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _log_event(self, event_type: str, data: Dict):
        """Internal method to log an event with integrity check"""
        with self.lock:
            # Create event record
            event = {
                'timestamp': datetime.now().isoformat(),
                'event_type': event_type,
                'data': data
            }
            
            # Convert to JSON
            event_json = json.dumps(event, ensure_ascii=False)
            
            # Calculate hash if integrity checking enabled
            if self.enable_integrity_check:
                event_hash = self._calculate_hash(event_json)
                event['integrity_hash'] = event_hash
                event_json = json.dumps(event, ensure_ascii=False)
                self.last_hash = event_hash
                self._save_hash(event_hash)
            
            # Log the event
            self.logger.info(event_json)
    
    def log_system_access(self, user: str, action: str, details: Optional[Dict] = None):
        """
        Log system access event
        
        Args:
            user: User identifier
            action: Action performed
            details: Additional details
        """
        data = {
            'user': user,
            'action': action
        }
        if details:
            data.update(details)
        
        self._log_event('system_access', data)
    
    def log_query(self, query: str, user: Optional[str] = None, 
                  results_count: Optional[int] = None):
        """
        Log a search query
        
        Args:
            query: Query text
            user: User who made the query
            results_count: Number of results returned
        """
        data = {
            'query': query[:200],  # Limit query length in logs
            'query_length': len(query)
        }
        if user:
            data['user'] = user
        if results_count is not None:
            data['results_count'] = results_count
        
        self._log_event('query', data)
    
    def log_model_invocation(self, model_name: str, 
                            inference_time: float,
                            tokens_generated: Optional[int] = None,
                            memory_usage: Optional[float] = None):
        """
        Log LLM model invocation
        
        Args:
            model_name: Name of the model
            inference_time: Time taken for inference
            tokens_generated: Number of tokens generated
            memory_usage: Memory usage in MB
        """
        data = {
            'model_name': model_name,
            'inference_time': inference_time
        }
        if tokens_generated is not None:
            data['tokens_generated'] = tokens_generated
        if memory_usage is not None:
            data['memory_usage_mb'] = memory_usage
        
        self._log_event('model_invocation', data)
    
    def log_retrieval(self, query: str, top_k: int, 
                     retrieval_time: float,
                     num_results: int):
        """
        Log retrieval operation
        
        Args:
            query: Query text
            top_k: Number of results requested
            retrieval_time: Time taken for retrieval
            num_results: Number of results returned
        """
        data = {
            'query_preview': query[:100],
            'top_k': top_k,
            'retrieval_time': retrieval_time,
            'num_results': num_results
        }
        
        self._log_event('retrieval', data)
    
    def log_error(self, error_type: str, error_message: str, 
                  context: Optional[Dict] = None):
        """
        Log an error event
        
        Args:
            error_type: Type of error
            error_message: Error message
            context: Additional context
        """
        data = {
            'error_type': error_type,
            'error_message': error_message
        }
        if context:
            data.update(context)
        
        self._log_event('error', data)
    
    def verify_integrity(self, log_file_path: str) -> bool:
        """
        Verify integrity of log file
        
        Args:
            log_file_path: Path to log file
            
        Returns:
            True if integrity check passes
        """
        if not self.enable_integrity_check:
            self.logger.warning("Integrity checking not enabled")
            return True
        
        try:
            with open(log_file_path, 'r') as f:
                lines = f.readlines()
            
            current_hash = None
            
            for line in lines:
                if not line.strip():
                    continue
                
                try:
                    # Parse log entry
                    parts = line.split(' - ')
                    if len(parts) < 4:
                        continue
                    
                    # Extract JSON part
                    json_part = ' - '.join(parts[3:])
                    event = json.loads(json_part)
                    
                    if 'integrity_hash' not in event:
                        continue
                    
                    # Verify hash
                    stored_hash = event.pop('integrity_hash')
                    event_json = json.dumps(event, ensure_ascii=False)
                    
                    if current_hash:
                        expected_hash = hashlib.sha256((current_hash + event_json).encode()).hexdigest()
                    else:
                        expected_hash = hashlib.sha256(event_json.encode()).hexdigest()
                    
                    if stored_hash != expected_hash:
                        self.logger.error(f"Integrity check failed at line: {line[:100]}")
                        return False
                    
                    current_hash = stored_hash
                
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    self.logger.warning(f"Error verifying line: {e}")
                    continue
            
            self.logger.info("Integrity check passed")
            return True
        
        except Exception as e:
            self.logger.error(f"Integrity verification failed: {e}")
            return False
    
    def get_statistics(self) -> Dict:
        """
        Get statistics from audit logs
        
        Returns:
            Dict with statistics
        """
        stats = {
            'total_events': 0,
            'event_types': {},
            'total_queries': 0,
            'total_model_invocations': 0,
            'total_errors': 0
        }
        
        try:
            # Read all log files in directory
            for log_file in self.log_directory.glob('audit_*.log'):
                with open(log_file, 'r') as f:
                    for line in f:
                        if not line.strip():
                            continue
                        
                        try:
                            parts = line.split(' - ')
                            if len(parts) < 4:
                                continue
                            
                            json_part = ' - '.join(parts[3:])
                            event = json.loads(json_part)
                            
                            stats['total_events'] += 1
                            
                            event_type = event.get('event_type', 'unknown')
                            stats['event_types'][event_type] = stats['event_types'].get(event_type, 0) + 1
                            
                            if event_type == 'query':
                                stats['total_queries'] += 1
                            elif event_type == 'model_invocation':
                                stats['total_model_invocations'] += 1
                            elif event_type == 'error':
                                stats['total_errors'] += 1
                        
                        except Exception:
                            continue
        
        except Exception as e:
            self.logger.error(f"Error getting statistics: {e}")
        
        return stats
