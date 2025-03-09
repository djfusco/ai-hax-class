import serpapi

# Replace with your SerpApi key
API_KEY = "21eb3415d7e788e8d411d2917596356e8e975a2ca2ba099170a842907c416f30"

def get_top_videos_serpapi(topic, max_results=3):
    params = {
        "api_key": API_KEY,
        "engine": "youtube",
        "search_query": topic
    }
    
    # Attempt to use the serpapi search function directly
    results = serpapi.search(params)
    top_videos = results.get("video_results", [])[:max_results]
    
    for i, video in enumerate(top_videos, 1):
        print(f"{i}. {video['title']}")
        print(f"   URL: {video['link']}")
        print()

if __name__ == "__main__":
    topic = input("Enter a topic: ")
    get_top_videos_serpapi(topic)