"""
Exception hierarchy for Deforum Flux

This module provides a comprehensive exception hierarchy to replace the
scattered error handling identified in the audit.
"""

from typing import Optional, Dict, Any


class DeforumException(Exception):
    """Base exception for all Deforum-related errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None):
        """
        Initialize Deforum exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
            original_error: Optional original exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.original_error = original_error
    
    def __str__(self) -> str:
        """Return string representation of the exception."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} (Details: {details_str})"
        return self.message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for logging/serialization."""
        result = {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "details": self.details
        }
        if self.original_error:
            result["original_error"] = {
                "type": type(self.original_error).__name__,
                "message": str(self.original_error)
            }
        return result


class FluxModelError(DeforumException):
    """Errors related to Flux model loading, initialization, or inference."""
    
    def __init__(self, message: str, model_name: Optional[str] = None, 
                 device: Optional[str] = None, **kwargs):
        """
        Initialize Flux model error.
        
        Args:
            message: Error message
            model_name: Name of the Flux model that caused the error
            device: Device where the error occurred
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if model_name:
            details["model_name"] = model_name
        if device:
            details["device"] = device
        
        super().__init__(message, details)


class ModelLoadingError(FluxModelError):
    """Specific error for model loading failures."""
    
    def __init__(self, message: str, model_path: Optional[str] = None, **kwargs):
        """
        Initialize model loading error.
        
        Args:
            message: Error message
            model_path: Path to the model that failed to load
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if model_path:
            details["model_path"] = model_path
        
        super().__init__(message, **details)


class DeforumConfigError(DeforumException):
    """Errors related to configuration validation or processing."""
    
    def __init__(self, message: str, config_field: Optional[str] = None, 
                 config_value: Optional[Any] = None, **kwargs):
        """
        Initialize configuration error.
        
        Args:
            message: Error message
            config_field: Name of the configuration field that caused the error
            config_value: Value that caused the error
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if config_field:
            details["config_field"] = config_field
        if config_value is not None:
            details["config_value"] = str(config_value)
        
        super().__init__(message, details)


class ValidationError(DeforumException):
    """Errors related to input validation."""
    
    def __init__(self, message: str, validation_errors: Optional[list] = None,
                 field_name: Optional[str] = None, **kwargs):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            validation_errors: List of specific validation errors
            field_name: Name of the field that failed validation
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if validation_errors:
            details["validation_errors"] = validation_errors
        if field_name:
            details["field_name"] = field_name
        
        super().__init__(message, details)


class ParameterError(DeforumException):
    """Errors related to parameter parsing or processing."""
    
    def __init__(self, message: str, parameter_name: Optional[str] = None,
                 parameter_value: Optional[Any] = None, **kwargs):
        """
        Initialize parameter error.
        
        Args:
            message: Error message
            parameter_name: Name of the parameter that caused the error
            parameter_value: Value that caused the error
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if parameter_name:
            details["parameter_name"] = parameter_name
        if parameter_value is not None:
            details["parameter_value"] = str(parameter_value)
        
        super().__init__(message, details)


class MotionProcessingError(DeforumException):
    """Errors related to motion processing and animation generation."""
    
    def __init__(self, message: str, frame_index: Optional[int] = None,
                 motion_params: Optional[Dict[str, Any]] = None, original_error: Optional[Exception] = None, **kwargs):
        """
        Initialize motion processing error.
        
        Args:
            message: Error message
            frame_index: Index of the frame where the error occurred
            motion_params: Motion parameters that caused the error
            original_error: Optional original exception that caused this error
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if frame_index is not None:
            details["frame_index"] = frame_index
        if motion_params:
            details["motion_params"] = motion_params
        
        super().__init__(message, details, original_error)


class TensorProcessingError(DeforumException):
    """Errors related to tensor operations and processing."""
    
    def __init__(self, message: str, tensor_shape: Optional[tuple] = None,
                 expected_shape: Optional[tuple] = None, **kwargs):
        """
        Initialize tensor processing error.
        
        Args:
            message: Error message
            tensor_shape: Actual tensor shape that caused the error
            expected_shape: Expected tensor shape
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if tensor_shape:
            details["tensor_shape"] = tensor_shape
        if expected_shape:
            details["expected_shape"] = expected_shape
        
        super().__init__(message, details)


class ResourceError(DeforumException):
    """Errors related to system resources (memory, disk, etc.)."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 available: Optional[str] = None, required: Optional[str] = None, **kwargs):
        """
        Initialize resource error.
        
        Args:
            message: Error message
            resource_type: Type of resource (memory, disk, etc.)
            available: Available resource amount
            required: Required resource amount
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if resource_type:
            details["resource_type"] = resource_type
        if available:
            details["available"] = available
        if required:
            details["required"] = required
        
        super().__init__(message, details)


class DeforumTimeoutError(DeforumException):
    """Errors related to operation timeouts."""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None,
                 operation: Optional[str] = None, **kwargs):
        """
        Initialize timeout error.

        Args:
            message: Error message
            timeout_seconds: Timeout duration in seconds
            operation: Name of the operation that timed out
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if timeout_seconds:
            details["timeout_seconds"] = timeout_seconds
        if operation:
            details["operation"] = operation

        super().__init__(message, details)


class APIError(DeforumException):
    """Errors related to API calls and external services."""
    
    def __init__(self, message: str, status_code: Optional[int] = None,
                 endpoint: Optional[str] = None, **kwargs):
        """
        Initialize API error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            endpoint: API endpoint that caused the error
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if status_code:
            details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint
        
        super().__init__(message, details)


# Exception mapping for common error patterns
EXCEPTION_MAPPING = {
    "model_loading": ModelLoadingError,
    "flux_model": FluxModelError,
    "config": DeforumConfigError,
    "validation": ValidationError,
    "parameter": ParameterError,
    "motion": MotionProcessingError,
    "tensor": TensorProcessingError,
    "resource": ResourceError,
    "timeout": DeforumTimeoutError,
    "api": APIError
}


def create_exception(error_type: str, message: str, **kwargs) -> DeforumException:
    """
    Create an exception of the appropriate type.
    
    Args:
        error_type: Type of error (key in EXCEPTION_MAPPING)
        message: Error message
        **kwargs: Additional error details
        
    Returns:
        Appropriate exception instance
    """
    exception_class = EXCEPTION_MAPPING.get(error_type, DeforumException)
    return exception_class(message, **kwargs)


def handle_exception(func):
    """
    Decorator to handle exceptions and convert them to appropriate Deforum exceptions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except DeforumException:
            # Re-raise Deforum exceptions as-is
            raise
        except FileNotFoundError as e:
            raise DeforumConfigError(f"File not found: {e}", file_path=str(e))
        except ValueError as e:
            raise ValidationError(f"Invalid value: {e}")
        except RuntimeError as e:
            if "CUDA" in str(e) or "GPU" in str(e):
                raise FluxModelError(f"GPU/CUDA error: {e}")
            raise DeforumException(f"Runtime error: {e}")
        except MemoryError as e:
            raise ResourceError(f"Out of memory: {e}", resource_type="memory")
        except KeyError as e:
            # Handle specific KeyError issues (like LogRecord problems)
            if "module" in str(e) and "LogRecord" in str(type(e)):
                raise DeforumException(f"Logging configuration error: {e}", details={"original_exception": type(e)})
            raise ValidationError(f"Missing key: {e}")
        except Exception as e:
            # More detailed error information
            error_details = {
                "original_exception": type(e).__name__,
                "error_message": str(e),
                "function": func.__name__ if hasattr(func, '__name__') else 'unknown'
            }
            raise DeforumException(f"Unexpected error: \"{e}\"", details=error_details)
    
    return wrapper 

class SecurityError(DeforumException):
    """Errors related to security violations and input validation."""
    
    def __init__(self, message: str, security_violation: Optional[str] = None,
                 input_value: Optional[str] = None, **kwargs):
        """
        Initialize security error.
        
        Args:
            message: Error message
            security_violation: Type of security violation
            input_value: Input that caused the security violation
            **kwargs: Additional error details
        """
        details = kwargs.copy()
        if security_violation:
            details["security_violation"] = security_violation
        if input_value:
            details["input_value"] = str(input_value)
        
        super().__init__(message, details)


# Update the exception mapping
EXCEPTION_MAPPING.update({
    "security": SecurityError
})
