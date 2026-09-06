"""
InsightAI - Input validators & safety helpers.

The application never executes or deserialises uploaded content as code. These
helpers guard against unsafe file names, unsupported formats, oversized
uploads and empty datasets. They raise :class:`InsightAIError` subclasses that
the UI turns into friendly warnings instead of crashes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- Errors


class InsightAIError(Exception):
    """Base error for all user-facing InsightAI failures."""


class UnsupportedFormatError(InsightAIError):
    """Raised when the uploaded file type is not supported."""


class EmptyDatasetError(InsightAIError):
    """Raised when an uploaded dataset contains no rows/columns."""


class OversizeError(InsightAIError):
    """Raised when the upload exceeds the configured size limit."""


class CorruptFileError(InsightAIError):
    """Raised when the file cannot be parsed."""


# --------------------------------------------------------------------------- Rules

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".tsv", ".parquet"}

# Characters that are unsafe in filenames on Windows / POSIX.
_UNSAFE_CHARS = set('<>:"/\\|?*\0')
_UNSAFE_NAMES = {"con", "prn", "aux", "nul", "com1", "com2", "com3", "com4",
                 "com5", "com6", "com7", "com8", "com9",
                 "lpt1", "lpt2", "lpt3", "lpt4", "lpt5", "lpt6", "lpt7", "lpt8", "lpt9"}


def sanitize_filename(filename: str, fallback: str = "dataset") -> str:
    """Return a filesystem-safe version of ``filename``."""
    if not filename:
        return fallback
    name = Path(filename).name  # strip any path components
    name = "".join("_" if ch in _UNSAFE_CHARS else ch for ch in name)
    stem = Path(name).stem.strip().lower()
    if not stem or stem in _UNSAFE_NAMES:
        name = f"{fallback}_{name}"
    # Guard against trailing dots/spaces.
    return name.rstrip(" .") or fallback


def validate_extension(filename: str) -> str:
    """Validate the file extension and return it lower-cased."""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFormatError(
            f"Unsupported file type '{suffix or 'unknown'}'. "
            f"Supported types: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        )
    return suffix


def validate_upload_size(file_size_bytes: int, max_mb: int) -> None:
    """Raise :class:`OversizeError` if ``file_size_bytes`` exceeds the limit."""
    limit = max_mb * 1024 * 1024
    if file_size_bytes > limit:
        raise OversizeError(
            f"File is {file_size_bytes / (1024 * 1024):.1f} MB which exceeds the "
            f"configured {max_mb} MB limit."
        )


def validate_dataframe_shape(rows: int, cols: int) -> None:
    """Raise :class:`EmptyDatasetError` when the frame is empty."""
    if rows == 0 or cols == 0:
        raise EmptyDatasetError(
            "The dataset contains no usable rows/columns. Please upload a non-empty file."
        )
