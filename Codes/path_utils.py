import hashlib
import re


_INVALID_PATH_CHARS = re.compile(r'[^A-Za-z0-9._-]+')
_WINDOWS_RESERVED_NAMES = {
    'CON', 'PRN', 'AUX', 'NUL',
    *(f'COM{i}' for i in range(1, 10)),
    *(f'LPT{i}' for i in range(1, 10)),
}


def safe_path_component(value, fallback='run', max_length=120):
    """Return a stable, filesystem-safe component without losing uniqueness."""
    raw = str(value).strip() if value is not None else ''
    if not raw:
        raw = str(fallback).strip() or 'run'

    sanitized = _INVALID_PATH_CHARS.sub('_', raw).strip(' .') or 'run'
    changed = sanitized != raw
    if sanitized.upper() in _WINDOWS_RESERVED_NAMES:
        sanitized = f'_{sanitized}'
        changed = True

    digest = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:10]
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length - 11].rstrip(' ._-')
        changed = True
    if changed:
        sanitized = f'{sanitized}_{digest}'
    return sanitized
