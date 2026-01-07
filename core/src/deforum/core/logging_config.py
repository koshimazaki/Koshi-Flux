"""
Logging configuration for Deforum Flux

This module provides centralized logging configuration with performance monitoring,
structured logging, and multiple output formats.
"""

import logging
import logging.handlers
import sys
import time
import functools
import json
from pathlib import Path
from typing import Optional, Dict, Any, Union
from datetime import datetime


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records."""
    
    def filter(self, record):
        """Add performance information to log record."""
        record.timestamp = time.time()
        record.iso_timestamp = datetime.now().isoformat()
        return True


class StructuredFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs."""
    
    def format(self, record):
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": getattr(record, "iso_timestamp", datetime.now().isoformat()),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "source_module": getattr(record, "filename", "unknown"),  # Fixed: Use filename instead of module
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields (avoid overwriting core fields)
        reserved_keys = {
            "name", "msg", "args", "levelname", "levelno", "pathname", 
            "filename", "module", "lineno", "funcName", "created", 
            "msecs", "relativeCreated", "thread", "threadName", 
            "processName", "process", "getMessage", "exc_info", 
            "exc_text", "stack_info", "timestamp", "iso_timestamp",
            "level", "logger", "message", "source_module", "function", "line"
        }
        
        for key, value in record.__dict__.items():
            if key not in reserved_keys:
                # Use the key directly if it doesn't conflict
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'       # Reset
    }
    
    def format(self, record):
        """Format log record with colors."""
        log_color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset_color = self.COLORS['RESET']
        
        # Create colored level name
        colored_levelname = f"{log_color}{record.levelname}{reset_color}"
        
        # Format the message
        formatted_message = super().format(record)
        
        # Replace levelname with colored version
        formatted_message = formatted_message.replace(record.levelname, colored_levelname)
        
        return formatted_message


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    console_output: bool = True,
    structured_logging: bool = False,
    enable_performance_logging: bool = True,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Set up comprehensive logging configuration.
    
    Args:
        level: Logging level (string or int)
        log_file: Path to log file (optional)
        console_output: Whether to output to console
        structured_logging: Whether to use structured JSON logging
        enable_performance_logging: Whether to enable performance logging
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
        
    Returns:
        Configured root logger
    """
    # Convert string level to int if needed
    if isinstance(level, str):
        level = getattr(logging, level.upper())
    
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Add performance filter if enabled
    if enable_performance_logging:
        perf_filter = PerformanceFilter()
        root_logger.addFilter(perf_filter)
    
    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        
        if structured_logging:
            console_formatter = StructuredFormatter()
        else:
            console_formatter = ColoredFormatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use rotating file handler
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count
        )
        file_handler.setLevel(level)
        
        if structured_logging:
            file_formatter = StructuredFormatter()
        else:
            file_formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
    
    # Log the setup
    setup_logger = logging.getLogger(__name__)
    setup_logger.info(f"Logging initialized - Level: {logging.getLevelName(level)}")
    if log_file:
        setup_logger.info(f"Log file: {log_file}")
    if structured_logging:
        setup_logger.info("Structured JSON logging enabled")
    if enable_performance_logging:
        setup_logger.info("Performance logging enabled")
    
    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_performance(func):
    """
    Decorator to log function performance metrics.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        # Log function start
        logger.debug(f"Starting {func.__name__}", extra={
            "function": func.__name__,
            "source_module": func.__module__,
            "args_count": len(args),
            "kwargs_count": len(kwargs)
        })
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful completion
            logger.info(f"Completed {func.__name__} in {execution_time:.3f}s", extra={
                "function": func.__name__,
                "source_module": func.__module__,
                "execution_time": execution_time,
                "status": "success"
            })
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Log error
            logger.error(f"Error in {func.__name__} after {execution_time:.3f}s: {e}", extra={
                "function": func.__name__,
                "source_module": func.__module__,
                "execution_time": execution_time,
                "status": "error",
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            
            raise
    
    return wrapper


def log_memory_usage(func):
    """
    Decorator to log memory usage of functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        
        try:
            import psutil
            process = psutil.Process()
            
            # Get memory before
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            # Get memory after
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            mem_delta = mem_after - mem_before
            
            logger.debug(f"Memory usage for {func.__name__}: {mem_before:.1f}MB -> {mem_after:.1f}MB (Δ{mem_delta:+.1f}MB)", extra={
                "function": func.__name__,
                "memory_before_mb": mem_before,
                "memory_after_mb": mem_after,
                "memory_delta_mb": mem_delta
            })
            
            return result
            
        except ImportError:
            # psutil not available, just run the function
            logger.debug(f"Memory logging unavailable for {func.__name__} (psutil not installed)")
            return func(*args, **kwargs)
    
    return wrapper


class LogContext:
    """Context manager for structured logging with additional context."""
    
    def __init__(self, logger: logging.Logger, operation: str, **context):
        """
        Initialize log context.
        
        Args:
            logger: Logger to use
            operation: Name of the operation
            **context: Additional context to include in logs
        """
        self.logger = logger
        self.operation = operation
        self.context = context
        self.start_time = None
    
    def __enter__(self):
        """Enter the context."""
        self.start_time = time.time()
        self.logger.info(f"Starting {self.operation}", extra=self.context)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context."""
        execution_time = time.time() - self.start_time if self.start_time else 0
        
        context = self.context.copy()
        context["execution_time"] = execution_time
        
        if exc_type is None:
            context["status"] = "success"
            self.logger.info(f"Completed {self.operation} in {execution_time:.3f}s", extra=context)
        else:
            context["status"] = "error"
            context["error_type"] = exc_type.__name__
            context["error_message"] = str(exc_val)
            self.logger.error(f"Failed {self.operation} after {execution_time:.3f}s: {exc_val}", extra=context)
    
    def log(self, message: str, level: int = logging.INFO, **extra_context):
        """Log a message with the current context."""
        context = self.context.copy()
        context.update(extra_context)
        self.logger.log(level, message, extra=context)


# Pre-configured logger instances
def get_bridge_logger() -> logging.Logger:
    """Get logger for bridge operations."""
    return get_logger("deforum.bridge")


def get_model_logger() -> logging.Logger:
    """Get logger for model operations."""
    return get_logger("deforum.model")


def get_motion_logger() -> logging.Logger:
    """Get logger for motion processing."""
    return get_logger("deforum.motion")


def get_config_logger() -> logging.Logger:
    """Get logger for configuration operations."""
    return get_logger("deforum.config")


# Example usage patterns
if __name__ == "__main__":
    # Example setup
    setup_logging(
        level="INFO",
        log_file="deforum.log",
        structured_logging=True,
        enable_performance_logging=True
    )
    
    test_logger = get_logger(__name__)
    
    # Example usage
    with LogContext(test_logger, "test_operation", user_id="test", operation_type="example"):
        test_logger.info("This is a test message")
        time.sleep(1)  # Simulate work
    
    # Example decorated function
    @log_performance
    @log_memory_usage
    def example_function():
        test_logger.info("Doing some work...")
        time.sleep(0.5)
        return "result"
    
    result = example_function()
    test_logger.info(f"Got result: {result}") 