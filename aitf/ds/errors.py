"""Custom exceptions for the datastore module."""


class DataStoreError(Exception):
    """Base exception for ds module."""


class GoldenNotFoundError(DataStoreError, KeyError):
    """Raised when golden data (model/version/operator) is not found."""
