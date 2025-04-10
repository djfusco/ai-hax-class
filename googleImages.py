import requests

api_key = ""
cse_id = ""
query = "ai in society"
url = f"https://www.googleapis.com/customsearch/v1?key={api_key}&cx={cse_id}&q={query}&searchType=image&rights=cc_publicdomain%20cc_attribute%20cc_sharealike"

response = requests.get(url)
data = response.json()

# Extract image results
items = data.get("items", [])
for item in items:
    print(f"Title: {item['title']}")
    print(f"Link: {item['link']}")
    print(f"Source: {item['displayLink']}")
    print("---")