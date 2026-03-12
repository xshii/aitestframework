"""Custom exceptions for the test-case module."""


class TcError(Exception):
    """Base exception for tc module."""


class ExecutionNotFoundError(TcError, KeyError):
    """Raised when an execution ID is not found."""


class SuiteLoadError(TcError):
    """Raised when test suites fail to load."""
