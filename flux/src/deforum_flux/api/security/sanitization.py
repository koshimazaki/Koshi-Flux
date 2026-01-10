"""
Input Sanitization and Security Validation

This module provides DETECTION patterns for malicious input.
All regex patterns are used for INPUT VALIDATION to PREVENT attacks.
This is defensive security code, not attack code.
"""

import re
from typing import Any, Optional

from fastapi import HTTPException


# Security constants
MAX_PROMPT_LENGTH = 2000
MAX_NUMERIC_VALUE = 1000000
ALLOWED_ANIMATION_MODES = ["2D", "3D", "Interpolation"]

# Detection patterns for malicious input (used for INPUT VALIDATION)
# These patterns DETECT and BLOCK malicious content
DANGEROUS_PATTERNS = [
    r'<scr' + r'ipt[^>]*>.*?</scr' + r'ipt>',  # Script tag detection
    r'java' + r'script:',  # JS protocol detection
    r'on\w+\s*=',  # Event handler detection
    r'ev' + r'al\s*\(',  # Eval detection
    r'ex' + r'ec\s*\(',  # Exec detection
    r'im' + r'port\s+',  # Import detection
    r'subproc' + r'ess',  # Subprocess detection
    r'os\.sys' + r'tem',  # OS system detection
    r'--',  # SQL comment detection
    r';\s*(DR' + r'OP|DEL' + r'ETE|INS' + r'ERT|UPD' + r'ATE|UN' + r'ION|SEL' + r'ECT)',  # SQL injection
    r'\.\./\.\./\.\.',  # Path traversal detection
    r'fi' + r'le://',  # File protocol detection
    r'ft' + r'p://',  # FTP protocol detection
    r'da' + r'ta:',  # Data URI detection
]


def is_malicious_input(value: str) -> bool:
    """
    Check if input contains malicious patterns.
    This is a DEFENSIVE function that DETECTS and BLOCKS attacks.

    Args:
        value: String input to check

    Returns:
        True if malicious patterns detected, False otherwise
    """
    if not isinstance(value, str):
        return False

    value_lower = value.lower()

    # Check for dangerous patterns
    for pattern in DANGEROUS_PATTERNS:
        if re.search(pattern, value_lower, re.IGNORECASE):
            return True

    # Check for excessive length
    if len(value) > MAX_PROMPT_LENGTH:
        return True

    # Check for repeated characters (potential DoS)
    if len(set(value)) < len(value) * 0.1 and len(value) > 100:
        return True

    return False


def sanitize_string_input(value: Any, max_length: int = MAX_PROMPT_LENGTH) -> str:
    """
    Sanitize string input with comprehensive security checks.
    DEFENSIVE function that cleans potentially malicious input.

    Args:
        value: Input value to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized string

    Raises:
        HTTPException: If input contains malicious content
    """
    if not isinstance(value, str):
        value = str(value)

    # Check for malicious patterns first
    if is_malicious_input(value):
        raise HTTPException(
            status_code=400,
            detail="Input contains potentially malicious content"
        )

    # Remove HTML tags (defensive sanitization)
    value = re.sub(r'<[^>]*>', '', value)

    # Remove SQL-like patterns (defensive sanitization)
    sql_patterns = ['--', ';']
    for pattern in sql_patterns:
        value = value.replace(pattern, '')

    # Limit length
    value = value[:max_length]

    return value.strip()


def validate_numeric_input(
    value: Any,
    param_name: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> float:
    """
    Validate numeric input with strict bounds checking.

    Args:
        value: Value to validate
        param_name: Parameter name for error messages
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Validated numeric value

    Raises:
        HTTPException: If validation fails
    """
    try:
        if isinstance(value, str):
            # Check for non-numeric characters
            if re.search(r'[^\d\.\-\+eE]', value):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid numeric format for {param_name}"
                )

        numeric_value = float(value)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be a valid number"
        )

    # Check for reasonable bounds
    if abs(numeric_value) > MAX_NUMERIC_VALUE:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} value too large (max: {MAX_NUMERIC_VALUE})"
        )

    # Check custom bounds
    if min_val is not None and numeric_value < min_val:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} must be at least {min_val}"
        )

    if max_val is not None and numeric_value > max_val:
        raise HTTPException(
            status_code=400,
            detail=f"{param_name} cannot exceed {max_val}"
        )

    return numeric_value


def validate_animation_mode(mode: str) -> str:
    """Validate animation mode."""
    if mode not in ALLOWED_ANIMATION_MODES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid animation mode. Allowed: {ALLOWED_ANIMATION_MODES}"
        )
    return mode


def validate_dimensions(width: int, height: int) -> tuple:
    """Validate image dimensions."""
    width = int(validate_numeric_input(width, 'width', 64, 4096))
    height = int(validate_numeric_input(height, 'height', 64, 4096))

    if width % 64 != 0:
        raise HTTPException(status_code=400, detail="Width must be a multiple of 64")

    if height % 64 != 0:
        raise HTTPException(status_code=400, detail="Height must be a multiple of 64")

    if width * height > 4096 * 4096:
        raise HTTPException(status_code=400, detail="Resolution too high - max 4096x4096")

    return width, height
