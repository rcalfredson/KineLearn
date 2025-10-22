from pathlib import Path


def find_unique(base_dir, patterns, must_contain=None):
    """Return the (first) unique Path matching the list of glob patterns."""
    candidates = []
    for pat in patterns:
        candidates.extend(base_dir.glob(pat))
    if must_contain:
        candidates = [c for c in candidates if must_contain in c.name]
    if not candidates:
        return None
    if len(candidates) > 1:
        print(f"[WARN] Multiple matches for {patterns!r}; using {candidates[0]}")
    return candidates[0]
