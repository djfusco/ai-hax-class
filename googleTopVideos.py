from googleapiclient.discovery import build
from datetime import datetime, timedelta
import re

Google_API_KEY=os.getenv("Google_API_KEY")


def get_top_educational_videos(topic, max_results=3, min_views=1000):
    # Initialize the YouTube API client
    youtube = build("youtube", "v3", developerKey=Google_API_KEY)
    
    # Search for videos by topic, prioritizing educational content
    search_response = youtube.search().list(
        q=f"{topic} tutorial explain concept",  # Bias toward educational content
        part="id,snippet",
        maxResults=20,  # Fetch more videos to analyze
        type="video",
        videoDefinition="high",  # Prefer higher quality videos
        relevanceLanguage="en"  # English results (change as needed)
    ).execute()
    
    # Extract video IDs and channels
    video_ids = [item["id"]["videoId"] for item in search_response["items"]]
    channel_ids = list(set([item["snippet"]["channelId"] for item in search_response["items"]]))
    
    # Get channel statistics for education signals
    channels_response = youtube.channels().list(
        part="statistics,snippet",
        id=",".join(channel_ids)
    ).execute()
    
    # Create channel info dictionary
    channel_info = {}
    for item in channels_response["items"]:
        channel_id = item["id"]
        subscriber_count = int(item["statistics"].get("subscriberCount", 0))
        video_count = int(item["statistics"].get("videoCount", 0))
        channel_title = item["snippet"]["title"]
        channel_info[channel_id] = {
            "subscriber_count": subscriber_count,
            "video_count": video_count,
            "channel_title": channel_title
        }
    
    # Fetch video details
    video_response = youtube.videos().list(
        part="snippet,statistics,contentDetails",
        id=",".join(video_ids)
    ).execute()
    
    # Fetch comments for engagement analysis
    comments_data = {}
    for video_id in video_ids:
        try:
            comments_response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=10,
                order="relevance"
            ).execute()
            
            if "items" in comments_response:
                comments_data[video_id] = comments_response["items"]
        except:
            # Video might have comments disabled
            comments_data[video_id] = []
    
    # Process videos with enhanced metrics
    videos = []
    educational_keywords = ["learn", "tutorial", "explained", "understand", "guide", 
                           "how to", "concept", "lesson", "course", "education"]
    
    for item in video_response["items"]:
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        video_id = item["id"]
        channel_id = item["snippet"]["channelId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        like_count = int(item["statistics"].get("likeCount", 0))
        view_count = int(item["statistics"].get("viewCount", 0))
        comment_count = int(item["statistics"].get("commentCount", 0))
        
        # Skip videos with too few views
        if view_count < min_views:
            continue
        
        # Calculate basic metrics
        like_view_ratio = (like_count / view_count) if view_count > 0 else 0
        engagement_ratio = ((like_count + comment_count) / view_count) if view_count > 0 else 0
        
        # Parse video duration
        duration = item["contentDetails"]["duration"]
        duration_seconds = parse_duration(duration)
        
        # Analyze publish date (recency factor)
        publish_date = datetime.strptime(item["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
        days_since_publish = (datetime.now() - publish_date).days
        recency_score = max(0, 1 - (days_since_publish / 365))  # Higher score for newer videos
        
        # Educational content signals
        educational_score = 0
        # Check title and description for educational keywords
        for keyword in educational_keywords:
            if keyword.lower() in title.lower():
                educational_score += 0.2
            if keyword.lower() in description.lower():
                educational_score += 0.1
        
        # Duration factor (penalize very short or very long videos)
        duration_score = 0
        if 300 <= duration_seconds <= 1200:  # 5-20 minutes is ideal for educational content
            duration_score = 1
        elif duration_seconds < 300:
            duration_score = duration_seconds / 300
        else:
            duration_score = max(0, 1 - ((duration_seconds - 1200) / 1800))
        
        # Channel authority score
        channel_data = channel_info.get(channel_id, {})
        subscriber_count = channel_data.get("subscriber_count", 0)
        video_count = channel_data.get("video_count", 0)
        
        channel_authority = min(1, (subscriber_count / 1000000) * 0.5 + (min(video_count, 100) / 100) * 0.5)
        
        # Comment quality analysis (looking for educational signals)
        comment_quality_score = 0
        if video_id in comments_data:
            comment_texts = []
            for comment_item in comments_data[video_id]:
                comment_text = comment_item["snippet"]["topLevelComment"]["snippet"]["textDisplay"].lower()
                comment_texts.append(comment_text)
            
            # Count educational keywords in comments
            edu_keyword_count = 0
            positive_signals = ["understand", "helped", "learned", "clear", "thank", "great explanation"]
            
            for comment in comment_texts:
                for signal in positive_signals:
                    if signal in comment:
                        edu_keyword_count += 1
            
            # Normalize by number of comments
            if len(comment_texts) > 0:
                comment_quality_score = min(1, edu_keyword_count / (len(comment_texts) * 2))
        
        # Calculate composite score (weighted for educational value)
        composite_score = (
            like_view_ratio * 0.3 +
            engagement_ratio * 0.15 +
            educational_score * 0.2 +
            duration_score * 0.1 +
            channel_authority * 0.15 +
            comment_quality_score * 0.05 +
            recency_score * 0.05
        )
        
        videos.append({
            "title": title,
            "url": url,
            "channel": channel_data.get("channel_title", "Unknown"),
            "like_count": like_count,
            "view_count": view_count,
            "comment_count": comment_count,
            "like_view_ratio": like_view_ratio,
            "duration": format_duration(duration_seconds),
            "published": publish_date.strftime("%Y-%m-%d"),
            "educational_score": educational_score,
            "engagement_score": engagement_ratio,
            "composite_score": composite_score
        })
    
    # Sort by composite score and take top results
    top_videos = sorted(
        videos,
        key=lambda x: x["composite_score"],
        reverse=True
    )[:max_results]
    
    return top_videos

def parse_duration(duration_str):
    """Convert ISO 8601 duration to seconds"""
    match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration_str)
    if not match:
        return 0
    
    hours = int(match.group(1)[:-1]) if match.group(1) else 0
    minutes = int(match.group(2)[:-1]) if match.group(2) else 0
    seconds = int(match.group(3)[:-1]) if match.group(3) else 0
    
    return hours * 3600 + minutes * 60 + seconds

def format_duration(seconds):
    """Format seconds as HH:MM:SS"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    if hours > 0:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes}:{seconds:02d}"

def main():
    topic = input("Enter a topic you want to learn about: ")
    print(f"\nSearching for the best educational videos about '{topic}'...")
    
    top_videos = get_top_educational_videos(topic)
    
    if not top_videos:
        print(f"No videos found for '{topic}' with sufficient views.")
        return
    
    print(f"\nTop {len(top_videos)} educational videos for '{topic}':\n")
    
    for i, video in enumerate(top_videos, 1):
        print(f"{i}. {video['title']}")
        print(f"   Channel: {video['channel']}")
        print(f"   URL: {video['url']}")
        print(f"   Likes: {video['like_count']:,}")
        print(f"   Views: {video['view_count']:,}")
        print(f"   Like-to-View Ratio: {video['like_view_ratio']:.2%}")
        print(f"   Duration: {video['duration']}")
        print(f"   Published: {video['published']}")
        print(f"   Educational Relevance Score: {video['educational_score']:.2f}")
        print(f"   Overall Quality Score: {video['composite_score']:.2f}")
        print()

if __name__ == "__main__":
    main()