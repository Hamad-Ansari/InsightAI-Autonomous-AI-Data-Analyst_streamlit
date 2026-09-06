import os
import requests

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")

print(f"Ollama URL: {BASE_URL}")
print(f"Chat model: {MODEL}")

try:
    r = requests.get(f"{BASE_URL}/api/tags", timeout=10)
    r.raise_for_status()
    data = r.json()
    names = [m.get("name") for m in data.get("models", [])]
    print("Installed models:")
    for name in names:
        print("  -", name)
    if MODEL not in names:
        print(f"\nWARNING: {MODEL!r} is not in the installed model list.")
        print("Run: ollama pull " + MODEL)
        raise SystemExit(2)

    payload = {"model": MODEL, "prompt": "Say hello in one short sentence.", "stream": False}
    r = requests.post(f"{BASE_URL}/api/generate", json=payload, timeout=120)
    r.raise_for_status()
    result = r.json()
    print("\nOllama generation test:")
    print(result.get("response", result))
    print("\nOllama API is working.")
except requests.exceptions.ConnectionError:
    print("\nERROR: Cannot connect to Ollama at", BASE_URL)
    print("Start Ollama Desktop or run: ollama serve")
    raise SystemExit(1)
except requests.HTTPError as e:
    print("\nERROR: Ollama returned HTTP", e.response.status_code)
    print(e.response.text)
    raise SystemExit(1)
except Exception as e:
    print("\nERROR:", e)
    raise SystemExit(1)
