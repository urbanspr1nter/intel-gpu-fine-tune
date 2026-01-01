import json

def validate_json_string(json_str: str) -> bool:
    try:
        json.loads(json_str)

        return True
    except:
        return False