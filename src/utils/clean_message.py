from utils.strip_think_tags import strip_think_tags

def clean_message(assistant_message: str) -> str:
  assistant_message = strip_think_tags(assistant_message).strip()

  if assistant_message.startswith("```json"):
    assistant_message = assistant_message[len("```json"):]

  if assistant_message.endswith("```"):
    assistant_message = assistant_message[:-len("```")]

  assistant_message = assistant_message.strip()

  return assistant_message
