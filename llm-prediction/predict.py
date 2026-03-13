import http.client
import json
import yaml
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("API_KEY")

conn = http.client.HTTPSConnection("chat.kiconnect.nrw")

input = """
model: Openai GPT OSS 120B
messages:
  - role: system
    content: You are a helpful assistant.
  - role: user
    content: Erzähle mir einen Witz
"""

payload = yaml.safe_load(input)
body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

headers = {
    'Content-Type': "application/json",
    'Authorization': "Bearer " + api_key
}

conn.request("POST", "/api/v1/chat/completions", body, headers)

res = conn.getresponse()
data = res.read()

print(data.decode("utf-8"))