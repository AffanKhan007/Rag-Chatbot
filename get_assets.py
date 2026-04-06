import urllib.request
import json

url = "https://api.github.com/repos/pgvector/pgvector/releases/latest"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
with urllib.request.urlopen(req) as response:
    data = json.loads(response.read().decode())
    for asset in data['assets']:
        if 'windows' in asset['name']:
            print(asset['name'], asset['browser_download_url'])
