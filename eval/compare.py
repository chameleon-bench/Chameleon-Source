"""
Result comparison utilities for cross-engine SQL evaluation.

Normalizes result rows from MySQL, PostgreSQL, and Oracle into a common
format, then performs order-insensitive multiset comparison with tolerant
value matching (numeric epsilon, string trailing-space, aggregate-string
reordering).
"""

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List


def normalize_value(val: Any) -> Any:
    """Normalize a single value for cross-engine comparison."""
    if val is None:
        return None
    if hasattr(val, 'read'):
        try:
            val = val.read()
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            return str(val)
        except Exception:
            pass
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, datetime):
        return val.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(val, date):
        return val.strftime('%Y-%m-%d')
    if isinstance(val, bytes):
        try:
            return val.decode('utf-8')
        except Exception:
            return str(val)
    if isinstance(val, (int, float)):
        return float(val)
    import datetime as _dt
    if isinstance(val, _dt.timedelta):
        return float(val.total_seconds())
    return str(val)


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a single result row (lowercase keys, normalize values)."""
    return {k.lower(): normalize_value(v) for k, v in row.items()}


_COMMON_SEPARATORS = [', ', ',']


def _normalize_agg_string(val: str) -> str:
    """Normalize aggregate strings (GROUP_CONCAT/STRING_AGG) by sorting sub-parts."""
    for sep in _COMMON_SEPARATORS:
        if sep in val:
            parts = [p.strip() for p in val.split(sep)]
            return sep.join(sorted(parts))
    return val


def _values_equal(a: Any, b: Any) -> bool:
    """Compare two values with tolerant matching."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == 0.0 and b == 0.0:
            return True
        return abs(a - b) <= max(1e-6, abs(a) * 1e-6)
    sa, sb = str(a).strip(), str(b).strip()
    if sa == sb:
        return True
    if _normalize_agg_string(sa) == _normalize_agg_string(sb):
        return True
    return False


def compare_results(
    source_rows: List[Dict[str, Any]],
    target_rows: List[Dict[str, Any]],
    source_label: str = 'Source',
    target_label: str = 'Target',
) -> Dict[str, Any]:
    """
    Compare source and target query results (order-insensitive multiset match).

    Returns:
        {
            'match': bool,
            'row_count_match': bool,
            'column_match': bool,
            'value_match': bool,
            'source_rows': int,
            'target_rows': int,
            'details': str,
        }
    """
    result = {
        'match': False,
        'row_count_match': False,
        'column_match': False,
        'value_match': False,
        'source_rows': len(source_rows),
        'target_rows': len(target_rows),
        'details': '',
    }

    if len(source_rows) != len(target_rows):
        result['details'] = (
            f"Row count mismatch: {source_label}={len(source_rows)}, "
            f"{target_label}={len(target_rows)}"
        )
        return result
    result['row_count_match'] = True

    if len(source_rows) == 0:
        result['match'] = True
        result['column_match'] = True
        result['value_match'] = True
        result['details'] = 'Both sides returned empty results'
        return result

    s_rows = [normalize_row(r) for r in source_rows]
    t_rows = [normalize_row(r) for r in target_rows]

    s_cols = set(s_rows[0].keys())
    t_cols = set(t_rows[0].keys())
    if s_cols != t_cols:
        result['details'] = (
            f"Column mismatch: {source_label}={sorted(s_cols)}, "
            f"{target_label}={sorted(t_cols)}"
        )
        common_cols = s_cols & t_cols
        if not common_cols:
            return result
        s_rows = [{k: v for k, v in r.items() if k in common_cols} for r in s_rows]
        t_rows = [{k: v for k, v in r.items() if k in common_cols} for r in t_rows]
    else:
        result['column_match'] = True

    def _row_to_sort_key(row: Dict[str, Any]) -> tuple:
        items = []
        for k in sorted(row.keys()):
            v = row[k]
            if v is None:
                items.append((k, 0, '', 0.0))
            elif isinstance(v, (int, float)):
                items.append((k, 1, '', v))
            else:
                items.append((k, 2, str(v), 0.0))
        return tuple(items)

    s_sorted = sorted(s_rows, key=_row_to_sort_key)
    t_sorted = sorted(t_rows, key=_row_to_sort_key)

    mismatches = []
    for i, (sr, tr) in enumerate(zip(s_sorted, t_sorted)):
        for col in sr:
            sv = sr.get(col)
            tv = tr.get(col)
            if not _values_equal(sv, tv):
                mismatches.append(
                    f"Row{i} col '{col}': {source_label}={sv!r}, {target_label}={tv!r}"
                )
                if len(mismatches) >= 5:
                    break
        if len(mismatches) >= 5:
            break

    if mismatches:
        result['details'] = f"Value mismatch ({len(mismatches)}): " + "; ".join(mismatches[:3])
        return result

    result['value_match'] = True
    result['match'] = True
    result['details'] = 'OK'
    if not result['column_match']:
        result['details'] = 'Column names differ but intersection values match'
    return result
