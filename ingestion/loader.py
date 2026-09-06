"""
InsightAI - Data ingestion engine.

Loads CSV / Excel / TSV / Parquet into a pandas DataFrame, handling encoding
(auto-detection over UTF-8, UTF-8-SIG and Latin-1), validating uploads, and
preserving the *original* frame so cleaning never mutates it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.validators import (
    CorruptFileError,
    EmptyDatasetError,
    UnsupportedFormatError,
    validate_extension,
    validate_upload_size,
    validate_dataframe_shape,
    sanitize_filename,
)

_ENCODINGS = ["utf-8-sig", "utf-8", "latin-1"]
_HAS_PYARROW = False
try:  # pragma: no cover - optional
    import pyarrow  # noqa: F401

    _HAS_PYARROW = True
except Exception:
    _HAS_PYARROW = False


def _read_text_table(path: Path, delimiter: Optional[str] = None) -> pd.DataFrame:
    """Read a delimited text file, trying multiple encodings."""
    last_error: Optional[Exception] = None
    for enc in _ENCODINGS:
        try:
            df = pd.read_csv(path, encoding=enc, sep=delimiter or ",", engine="python")
            if not df.empty:
                return df
        except Exception as exc:  # pragma: no cover - depends on file
            last_error = exc
    raise CorruptFileError(
        f"Could not read the text file. Last error: {last_error}"
    )


def load_dataset(
    path: str,
    *,
    max_mb: int = 200,
    # Optional per-file overrides for UI convenience.
    sheet_name: Optional[str] = 0,
    delimiter: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load a supported dataset file into a pandas DataFrame.

    Parameters
    ----------
    path:
        Path (or URL-like string) to the file on disk.
    max_mb:
        Upload size guard in megabytes.
    sheet_name:
        Sheet name / index for Excel files.
    delimiter:
        Delimiter for text files; auto-detected when ``None``.

    Returns
    -------
    pd.DataFrame
        The *original* dataset. Never mutated downstream.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    size = path.stat().st_size
    validate_upload_size(size, max_mb)

    suffix = validate_extension(path.name)
    df: Optional[pd.DataFrame] = None

    try:
        if suffix in {".csv", ".tsv", ".txt"}:
            delim = delimiter or ("\t" if suffix == ".tsv" else ",")
            df = _read_text_table(path, delim)
        elif suffix in {".xlsx", ".xls"}:
            if isinstance(sheet_name, int):
                df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
            else:
                df = pd.read_excel(path)
        elif suffix == ".parquet":
            if not _HAS_PYARROW:
                raise UnsupportedFormatError(
                    "Parquet support requires 'pyarrow'. Install it with "
                    "`pip install pyarrow`."
                )
            df = pd.read_parquet(path)
        else:
            raise UnsupportedFormatError(f"Unsupported file type: {suffix}")

    except UnsupportedFormatError:
        raise
    except EmptyDatasetError:
        raise
    except Exception as exc:
        raise CorruptFileError(
            f"Could not parse '{path.name}'. The file may be corrupted or in an "
            f"unsupported structure. Detail: {exc}"
        ) from exc

    if df is None:
        raise EmptyDatasetError("The file produced no data.")

    # Normalise column names to strings and drop fully-empty columns.
    df.columns = [str(c).strip() for c in df.columns]
    validate_dataframe_shape(df.shape[0], df.shape[1])
    return df


def sample_dataframe(df: pd.DataFrame, fraction: float, seed: int = 42) -> pd.DataFrame:
    """Return a random sample of ``fraction`` rows, preserving the original."""
    if fraction >= 1.0 or len(df) == 0:
        return df.copy()
    return df.sample(frac=min(max(fraction, 0.0), 1.0), random_state=seed)
