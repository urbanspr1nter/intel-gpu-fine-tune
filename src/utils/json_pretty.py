import json
from typing import Any

def prettify_json(unprettied_json: str) -> str:
    """
    Pretty-print a JSON document (2-space indent).

    Also recursively expands any *string values* that themselves contain JSON
    objects/arrays (e.g. a field whose value is "{\"a\":1}" or even
    "\"{\\\"a\\\":1}\"").

    Returns:
        A prettified JSON string.
    """
    def parse_embedded_container(s: str) -> Any:
        # Try to "unwrap" JSON that has been embedded as a string, possibly multiple times.
        # Only commit the conversion if we ultimately get a dict or list.
        current: Any = s
        for _ in range(10):  # safety cap to avoid pathological cases
            if not isinstance(current, str):
                break

            t = current.strip()
            if not t:
                break

            # Fast heuristic: only attempt JSON parsing if it could plausibly be JSON.
            # - '{' or '[' for objects/arrays
            # - '"' to allow double-encoded JSON (a JSON string containing JSON text)
            if t[0] not in '{["':
                break

            try:
                current = json.loads(t)
            except json.JSONDecodeError:
                break

        return current if isinstance(current, (dict, list)) else s

    def walk(v: Any) -> Any:
        if isinstance(v, dict):
            # Mutate in-place to preserve key insertion order
            for k, val in list(v.items()):
                v[k] = walk(val)
            return v

        if isinstance(v, list):
            return [walk(item) for item in v]

        if isinstance(v, str):
            parsed = parse_embedded_container(v)
            return walk(parsed) if isinstance(parsed, (dict, list)) else v

        return v

    try:
        root = json.loads(unprettied_json)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Input is not valid JSON: {e.msg} (line {e.lineno}, column {e.colno})"
        ) from e

    expanded = walk(root)
    return json.dumps(expanded, indent=2, ensure_ascii=False)
