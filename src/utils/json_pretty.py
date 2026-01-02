import json

def prettify_json(json_str) -> str:
    return json.dumps(json.loads(json_str), indent=2)