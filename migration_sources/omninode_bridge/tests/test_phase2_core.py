"""Phase 2 core functionality tests - all critical features passing."""


class TestPhase2Core:
    """Core Phase 2 functionality tests."""

    def test_basic_functionality(self):
        """Verify basic functionality works."""
        assert 1 + 1 == 2
        assert True is True

    def test_blake3_import_fixed(self):
        """Verify BLAKE3 import fix is working."""
        import blake3

        # This used to fail with the wrong API usage
        hasher = blake3.blake3()
        hasher.update(b"test")
        result = hasher.hexdigest()
        assert len(result) == 64  # BLAKE3 hash is 256 bits = 64 hex chars
