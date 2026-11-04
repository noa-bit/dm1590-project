"""Compatibility wrapper for the active test-data reader implementation.

The real implementation lives in utils.file_reader. Keeping this tiny shim
prevents the old root-level prototype from drifting out of sync with the code
used by main.py.
"""

from utils.file_reader import Result, TestDataManager

__all__ = ["Result", "TestDataManager"]
