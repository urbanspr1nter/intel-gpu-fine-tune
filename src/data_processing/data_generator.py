import openai
import json
import jsonlines
from concurrent.futures import Future, ThreadPoolExecutor

from utils.json_validate import validate_json_string
from utils.json_pretty import prettify_json
from utils.strip_think_tags import strip_think_tags

def get_prompt() -> str:
  prompt = r"""You are a data generator. Produce N examples of “invalid JSON” paired with the corrected “valid JSON”.

Output format (STRICT):
- Output must be ONLY a JSON array of objects. No prose.
- Each item must have:
  - "id" (string, unique like "ex001")
  - "invalid_json" (string; multi-line allowed)
  - "fixed_json" (string; multi-line; must be valid JSON)
  - "error_types" (array of strings)
  - "fixed_reason" (string; 1-2 sentences describing what was invalid and exactly what you changed)

Global rules:
- "fixed_json" must be formatted with 2-space indentation (pretty-printed).
- "invalid_json" MUST vary in formatting quality:
  - Sometimes nicely formatted
  - Often messy: single-line blobs, inconsistent indentation, random extra spaces, tabs, or awkward line breaks from rushed copy/paste
  - invalid_json indentation may be 0/1/3/4 spaces, tabs, or no indentation at all
- Each example must have 1-5 top-level keys.
- Keep booleans true/false, numbers, and null unquoted in the fixed JSON.
- Make fixes minimal: do not rename keys, reorder fields, or change meaning beyond validity (other than pretty-printing fixed_json).

Root-type preservation (IMPORTANT):
- The fixed_json MUST preserve the top-level JSON type of invalid_json.
  - If invalid_json begins with "[" (array), fixed_json MUST be an array (not wrapped in an object).
  - If invalid_json begins with "{" (object), fixed_json MUST be an object.
- Do not introduce wrapper keys like "items" unless the input already has them.

Apostrophe / punctuation realism mix (IMPORTANT):
- Prose-like strings must vary punctuation style across items:
  - Some examples use ASCII apostrophe ' (U+0027)
  - Some examples use curly apostrophe ’ (U+2019)
- Do not normalize punctuation in fixed_json; preserve exactly what appears in invalid_json (except required JSON escaping).
- Across each run of N=5:
  - At least 2 items must contain an ASCII apostrophe ' somewhere in invalid_json.
  - At least 1 item must contain a curly apostrophe ’ somewhere in invalid_json.
- In code/config/log-ish strings (paths, identifiers, env names, JSON-like fragments), prefer ASCII ' if any apostrophe appears.

Domains / content variety (IMPORTANT):
- The dataset must be diverse. Do NOT assume telemetry/logging only.
- Mix examples across domains, including:
  - speech/audio/video transcriptions (prose-like paragraphs, dialogue, filler words, timestamps)
  - meeting notes / summaries
  - metadata for media (title, speakers, chapters)
  - simple configs
  - occasional “system-ish” payloads (but not the majority)
- Many string values should be prose-like and long (sentences, punctuation, quotes, newlines).

Error types:
- Use one or more of:
  - "newline"
  - "quotes"
  - "backslash"
  - "unquoted_key"
  - "unquoted_value"
  - "comma"            (consecutive commas, trailing commas, or missing values in arrays)
  - "extra_brace"      (extra/mismatched braces/brackets)
  - "comment"          (// or /* */ comments)
  - "nonfinite_number" (NaN, Infinity, -Infinity)
- Each item's "error_types" must accurately reflect what is present in invalid_json.

IMPORTANT for "newline":
- When using the "newline" error type, the invalid_json MUST contain a literal line break inside a quoted string value (not an escaped "\n").
- The fixed_json MUST replace those literal line breaks with the two-character escape sequence "\n" inside the string.
- Do not remove content or join lines with spaces; preserve the exact text with "\n" inserted.

Guidance for "comment" cases (IMPORTANT):
- invalid_json may contain single-line (// ...) or block (/* ... */) comments.
- fixed_json MUST remove comments entirely (do not convert comments into new keys/fields).
- Do not add fields to preserve comment text; removing the comment is the minimal fix.

Guidance for "nonfinite_number" cases (IMPORTANT):
- JSON does not allow NaN, Infinity, or -Infinity as numbers.
- If invalid_json uses NaN/Infinity/-Infinity as unquoted values, fixed_json MUST preserve intent by converting them to strings:
  - NaN → "NaN"
  - Infinity → "Infinity"
  - -Infinity → "-Infinity"
- Do not replace them with null unless the invalid_json already implies null explicitly.

Comma / missing-value handling (IMPORTANT):
- JSON does not allow “empty elements” in arrays.
- Treat any missing value between separators as a "comma" error, even if whitespace/newlines appear between commas.
  - Examples of missing values you MUST fix by removing the empty slot:
    - [1,,2]
    - [1, ,2]
    - [1,
       ,
       2]
    - [1, /*comment*/, 2] (after removing comment, you may still have a missing value)
- A comma that appears where a value is expected (e.g., a comma-only “element line”) MUST be removed, and the surrounding commas adjusted so the array remains valid.
- Do not convert the comma into a value, and do not keep a standalone comma as an “item”.

Concrete examples of the kinds of fixes you must generate:

A) Newline inside a string (invalid) → escape with "\n" (fixed)
Invalid:
{"a":"this is one line
another line"}
Fixed:
{
  "a": "this is one line\nanother line"
}

B) Quotes inside a string (invalid) → escape as \" (fixed)
Invalid:
{"a":"this is invalid "because" of the quotes"}
Fixed:
{
  "a": "this is invalid \"because\" of the quotes"
}

C) Windows path backslashes (invalid) → double backslashes (fixed)
Invalid:
{ "path":"C:\Users\roger\Downloads\file.txt" }
Fixed:
{
  "path": "C:\\Users\\roger\\Downloads\\file.txt"
}

D) Unquoted keys (invalid) → quote keys (fixed)
Invalid:
{ a: "hello" }
Fixed:
{
  "a": "hello"
}

E) Unquoted string values (invalid) → quote those values (fixed)
Invalid:
{"env":prod,"owner":roger,"ok":true,"retries":3}
Fixed:
{
  "env": "prod",
  "owner": "roger",
  "ok": true,
  "retries": 3
}

F) Messy formatting example (invalid_json is poorly formatted + multiple errors)
Invalid (messy spacing, uneven indentation, tabs, and a literal newline inside the transcript string):
{session: {id: 42,
	"speaker": alice, "text":"okay so here is the plan
we do it tomorrow at 9", "ok":true}}
Fixed (pretty-printed, keys/values quoted, newline escaped):
{
  "session": {
    "id": 42,
    "speaker": "alice",
    "text": "okay so here is the plan\nwe do it tomorrow at 9",
    "ok": true
  }
}

G) Consecutive or trailing commas example
Invalid (consecutive + trailing):
{
"items": [1, "cat",, "dog",]
}
Fixed (pretty-printed, excess commas removed):
{
  "items": [
    1,
    "cat",
    "dog"
  ]
}

H) Non-finite numbers example (NaN/Infinity)
Invalid:
{"expiresInSec": Infinity, "latencyMs": NaN}
Fixed:
{
  "expiresInSec": "Infinity",
  "latencyMs": "NaN"
}

I) Comments example
Invalid:
{
  "ip": "203.0.113.42", // forwarded from edge
  "ok": true
}
Fixed:
{
  "ip": "203.0.113.42",
  "ok": true
}

J) Missing array value with whitespace between commas
Invalid:
["Hello", 3.14, , true]
Fixed:
[
  "Hello",
  3.14,
  true
]

K) Missing array value shown as a comma-only line
Invalid:
[
  "Hello",
  3.14,
  ,
  true
]
Fixed:
[
  "Hello",
  3.14,
  true
]

Guidance for unquoted_value cases (IMPORTANT):
- Barewords are ambiguous (could be intended string, boolean, null, or number).
- Use these rules:
  - If the token is exactly true/false/null → keep it unquoted.
  - If it is a valid JSON number literal → keep it unquoted.
  - If it is exactly NaN/Infinity/-Infinity → treat as nonfinite_number and quote it as a string.
  - Otherwise, treat it as a string and quote it.

Large-payload requirement:
- Generate N=5 items each run.
- EXACTLY 2 of the 5 items must be “large payloads”.
- A “large payload” means:
  - still only 1-5 top-level keys
  - but within those keys, include deep nesting and/or arrays totaling at least ~120 lines when pretty-printed in fixed_json (2-space indentation).
  - Large payloads should often be transcription-like (e.g., chapters array, speaker segments, long text blocks).
  - The invalid_json for large items should contain only 1–3 issues (surgical fix), not dozens, to discourage over-editing.
  - The invalid_json for large items should often be messy (partially pretty-printed, uneven indentation, or long lines).

Mix requirements for each run (N=5):
- Exactly 2 large payloads.
- The remaining 3 are small/medium payloads, with variety across domains.
- Across the 5 items, include at least 4 different error types overall.
- At least 3 of the 5 items should contain 2+ error types in the same example.
- At least 2 of the 5 items should include nested objects AND arrays (both).

Minimal-change guidance:
- Only apply the minimum edits needed to make valid JSON.
- Do not normalize content or “clean up” prose.
- Do not reorder keys, rename keys, or add/remove fields unless required to correct JSON validity.
- Preserve punctuation, capitalization, and wording exactly (except for required escaping/quoting/backslashes or removing comments).
- fixed_json should be pretty-printed, but invalid_json may be messy.

fixed_reason guidance:
- Be specific about what changed, e.g.:
  - “Escaped literal newlines inside the transcript text using \n and escaped inner quotes with \".”
  - “Quoted previously unquoted object keys and added quotes around bareword string values while leaving booleans/numbers unchanged.”
  - “Escaped a Windows path by doubling backslashes and pretty-printed the corrected JSON.”
  - “Removed JSON comments and converted non-finite numbers (NaN/Infinity) into strings.”
  - “Removed the empty array element represented by a separator-with-no-value (comma with whitespace) and pretty-printed the array.”

Now generate N=5.
Return ONLY the JSON array of objects.
"""

  return prompt

def create_client():
  return openai.Client(
    base_url="http://192.168.1.36:8000/v1",
    api_key="none"
  )

def generate(client: openai.Client, attempt=0):
  if attempt == 3:
    return []

  user_prompt = get_prompt()
  messages = [{"role": "user", "content": user_prompt}]
  model = "gpt-oss-20b"

  response = client.chat.completions.create(
    model=model,
    messages=messages,
    temperature=1.0,
    max_completion_tokens=8192,
    reasoning_effort="low" if model == "gpt-oss-20b" else "medium"
  )

  results = []
  try:
    assistant_message = response.choices[0].message.content
    
    assistant_message = strip_think_tags(assistant_message)

    if assistant_message.startswith("```json"):
      assistant_message = assistant_message[len("```json"):]

    if assistant_message.endswith("```"):
      assistant_message = assistant_message[:-len("```")]

    assistant_message = assistant_message.strip()

    generated_json = json.loads(assistant_message)

    print(len(generated_json))

    for g in generated_json:
      # Check fixed_json, not fixed_reason (which is just text)
      try:
        if not validate_json_string(prettify_json(g["fixed_json"])):
          print("Error: Invalid json in fixed_json!")
          continue
      except Exception as e:
        print(f"Validation error: {e}")
        continue

      results.append(
        {
          "invalid_json": g["invalid_json"],
          "fixed_json": g["fixed_json"],
          "fixed_reason": g["fixed_reason"]
        }
      )

    return results
  except:
    return generate(client, attempt + 1)



if __name__ == "__main__":
  client = create_client()


  results = []

  keep_going = True
  consecutive_failures = 0
  
  while keep_going:
    futures = []

    print(f"Number of examples: {len(results)}")
    with ThreadPoolExecutor(max_workers=2) as executor:
      future: Future = executor.submit(generate, client, 0)

      futures.append(future)

    batch_found = False
    for future in futures:
      batch = future.result()
      if batch:
        batch_found = True
        results.extend(batch)

    if not batch_found:
      consecutive_failures += 1
      print(f"Warning: Empty batch generated ({consecutive_failures}/5)")
      if consecutive_failures >= 5:
        print("Stopping due to repeated generation failures.")
        break
    else:
      consecutive_failures = 0

    with jsonlines.open("data.jsonl", "w") as j:
      j.write_all(results)

    if len(results) >= 1000:
      keep_going = False

  with jsonlines.open("data.jsonl", "w") as j:
    j.write_all(results)