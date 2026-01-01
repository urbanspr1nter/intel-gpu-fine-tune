import re

def strip_think_tags(message: str) -> str:
  if message.find("</think>") and not message.find("<think>"):
    return message.split("</think>")[1]

  message = re.sub(r'<think>.*?</think>', "", message, flags=re.DOTALL).strip()

  return message
