"""Tests for API security sanitization."""

import pytest

# Try to import security module - skip tests if not available
try:
    from fastapi import HTTPException
    from deforum_flux.api.security import (
        is_malicious_input,
        sanitize_string_input,
        validate_numeric_input,
        validate_animation_mode,
        validate_dimensions,
        ALLOWED_ANIMATION_MODES,
    )
    HAS_SECURITY = True
except (ImportError, ModuleNotFoundError):
    HAS_SECURITY = False
    HTTPException = Exception
    ALLOWED_ANIMATION_MODES = []


pytestmark = pytest.mark.skipif(
    not HAS_SECURITY,
    reason="deforum_flux.api.security module not available"
)


@pytest.mark.skipif(not HAS_SECURITY, reason="Security module not available")
class TestMaliciousInputDetection:
    """Tests for malicious input detection."""

    def test_safe_input_passes(self):
        """Normal prompts should pass."""
        assert not is_malicious_input("a beautiful sunset over mountains")
        assert not is_malicious_input("cyberpunk city at night")
        assert not is_malicious_input("portrait of a person, detailed")

    def test_sql_comment_detected(self):
        """SQL comment pattern should be detected."""
        assert is_malicious_input("test -- comment")

    def test_sql_injection_detected(self):
        """SQL injection patterns should be detected."""
        assert is_malicious_input("; DROP TABLE users")
        assert is_malicious_input("; SELECT * FROM users")

    def test_path_traversal_deep_detected(self):
        """Deep path traversal should be detected."""
        # Pattern requires 3+ levels
        traversal = ".." + "/" + ".." + "/" + ".." + "/etc/passwd"
        assert is_malicious_input(traversal)

    def test_script_injection_detected(self):
        """Script tag injection should be detected."""
        script = "<scr" + "ipt>alert(1)</scr" + "ipt>"
        assert is_malicious_input(script)

    def test_excessive_length_detected(self):
        """Excessive length should be detected."""
        long_string = "a" * 3000
        assert is_malicious_input(long_string)

    def test_empty_and_none_input(self):
        """Empty and None inputs should be safe."""
        assert not is_malicious_input("")
        assert not is_malicious_input(None)


@pytest.mark.skipif(not HAS_SECURITY, reason="Security module not available")
class TestStringSanitization:
    """Tests for string sanitization."""

    def test_normal_string_unchanged(self):
        """Normal strings should pass through."""
        result = sanitize_string_input("a beautiful landscape")
        assert result == "a beautiful landscape"

    def test_max_length_enforced(self):
        """Strings should be truncated to max length."""
        # String must be under 100 chars to avoid DoS detection (or have >10% unique)
        long_string = "abcdefghijklmnopqrstuvwxyz0123456789 " * 3  # ~111 chars, 37 unique
        result = sanitize_string_input(long_string, max_length=50)
        assert len(result) == 50

    def test_html_stripped(self):
        """HTML tags should be stripped."""
        result = sanitize_string_input("<b>bold</b> text")
        assert "<b>" not in result
        assert "bold" in result

    def test_sql_comment_is_rejected(self):
        """SQL comments should be rejected (malicious detection)."""
        # Note: sanitize_string_input rejects malicious patterns BEFORE sanitizing
        with pytest.raises(HTTPException) as exc_info:
            sanitize_string_input("test -- comment")
        assert exc_info.value.status_code == 400

    def test_malicious_raises_exception(self):
        """Malicious input should raise HTTPException."""
        script = "<scr" + "ipt>alert(1)</scr" + "ipt>"
        with pytest.raises(HTTPException) as exc_info:
            sanitize_string_input(script)
        assert exc_info.value.status_code == 400


@pytest.mark.skipif(not HAS_SECURITY, reason="Security module not available")
class TestNumericValidation:
    """Tests for numeric input validation."""

    def test_valid_value_passes(self):
        """Values in range should return the value."""
        assert validate_numeric_input(512, "width", 64, 4096) == 512.0
        assert validate_numeric_input(1024, "height", 64, 4096) == 1024.0

    def test_min_value_enforced(self):
        """Values below minimum should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_numeric_input(32, "width", 64, 4096)
        assert exc_info.value.status_code == 400
        assert "at least" in exc_info.value.detail

    def test_max_value_enforced(self):
        """Values above maximum should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_numeric_input(8192, "width", 64, 4096)
        assert exc_info.value.status_code == 400
        assert "exceed" in exc_info.value.detail

    def test_boundary_values(self):
        """Boundary values should pass."""
        assert validate_numeric_input(64, "width", 64, 4096) == 64.0
        assert validate_numeric_input(4096, "width", 64, 4096) == 4096.0


@pytest.mark.skipif(not HAS_SECURITY, reason="Security module not available")
class TestAnimationModeValidation:
    """Tests for animation mode validation."""

    def test_valid_modes_return_mode(self):
        """Valid animation modes should return the mode string."""
        for mode in ALLOWED_ANIMATION_MODES:
            result = validate_animation_mode(mode)
            assert result == mode

    def test_invalid_mode_raises_exception(self):
        """Invalid animation mode should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_animation_mode("invalid_mode")
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException):
            validate_animation_mode("4D")


@pytest.mark.skipif(not HAS_SECURITY, reason="Security module not available")
class TestDimensionValidation:
    """Tests for dimension validation."""

    def test_valid_dimensions_return_tuple(self):
        """Valid dimensions should return (width, height) tuple."""
        result = validate_dimensions(1024, 1024)
        assert result == (1024, 1024)

        result = validate_dimensions(512, 768)
        assert result == (512, 768)

    def test_non_multiple_of_64_raises(self):
        """Dimensions not multiple of 64 should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_dimensions(100, 128)
        assert "multiple of 64" in exc_info.value.detail

        with pytest.raises(HTTPException) as exc_info:
            validate_dimensions(1024, 100)
        assert "multiple of 64" in exc_info.value.detail

    def test_out_of_range_raises(self):
        """Out of range dimensions should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_dimensions(32, 1024)
        assert exc_info.value.status_code == 400

        with pytest.raises(HTTPException) as exc_info:
            validate_dimensions(1024, 8192)
        assert exc_info.value.status_code == 400
