import urllib.request
import json
from pprint import pprint

def fetch(url):
    print(f"Fetching {url}... ")
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=5.0) as response:
            data = json.loads(response.read().decode())
            print(f"Status Code: {response.status}")
            pprint(data)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    fetch("http://127.0.0.1:8000/metrics/health")
    fetch("http://127.0.0.1:8000/metrics/summary")
