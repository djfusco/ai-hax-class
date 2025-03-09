from qdrant_client import QdrantClient
from langchain_community.llms import Ollama
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import anthropic
import os
import json
from typing import List, Dict, Optional
import re
from dataclasses import dataclass
from rich.console import Console
from dotenv import load_dotenv
import uvicorn
import aiofiles
import pathlib
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from googleapiclient.discovery import build
from datetime import datetime, timedelta
from bs4 import BeautifulSoup  # For HTML parsing

load_dotenv()

app = FastAPI(
    title="Hax CLI Workflow API",
    description="API for converting natural language queries into Hax CLI commands and fetching educational YouTube videos",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = QdrantClient(url="http://localhost:5903")
ollama = Ollama(base_url="http://localhost:5904", model="mistral")
console = Console()
Google_API_KEY = os.getenv("GOOGLE_API_KEY")

@dataclass
class WorkflowResult:
    step: str
    status: str
    output: Optional[str] = None
    error: Optional[str] = None

class WorkflowRequest(BaseModel):
    query: str = Field(..., description="Natural language query describing what to create")

class WorkflowResponse(BaseModel):
    commands: List[str]
    descriptions: List[str]
    explanation: str
    confidence: float

class HaxCliRequest(BaseModel):
    query: str
    engine: str  # 'ICDS' or 'Claude'
    need_rag: Optional[bool] = False

class YouTubeVideoRequest(BaseModel):
    topic: str = Field(..., description="Topic to search for educational videos")
    max_results: Optional[int] = Field(3, description="Maximum number of videos to return")
    min_views: Optional[int] = Field(1000, description="Minimum view count for videos")

class YouTubeVideoResponse(BaseModel):
    videos: List[Dict]
    topic: str

class HaxSiteWorkflow:
    def __init__(self):
        self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def create_system_prompt(self):
        return """You are an AI assistant that interprets natural language queries into specific Hax CLI commands and generates content when requested. Assume commands are run from within the site directory, so no site name prefix is needed for paths.

        Available Workflows:
        1. Create a new website:
        - Command: hax site start --name [siteName] --y
        - Creates new directory with site
        - Opens localhost:3000

        2. Add new page(s) without content:
        - Command: hax site node:add --title [pageTitle] --y
        - Creates new page in the current site directory

        3. Add new page(s) with content:
        - Command: hax site node:add --title [pageTitle] --content [htmlContent] --y
        - Creates new page with provided content
        - If "about [topic]" is in the query, generate three paragraphs about the topic
        - If "with video" is in the query, include a placeholder for video content: <!-- VIDEO_PLACEHOLDER -->

        4. Add a child page under a parent page with content:
        - Command: hax site node:add --title [childPageTitle] --content [htmlContent] --y
        - Followed by: hax site node:edit --item-id [childId] --node-op parent --parent [parentId] --y

        5. Add a customized child page:
        - If customization is implied (e.g., "create a new page called 'black kittens from india' under 'black kittens'"),
          assume the child page requires customized content based on the parent.

        6. Update an existing page with video:
        - Workflow type: "update_page_with_video"
        - No hax command needed; the system will append video content to the existing page's index.html
        - Triggered by queries like "update the page called [title] with video"
        - Only specify the page title; video content will be fetched and appended separately

        Special Themes:
        - Penn State Theme: When request includes "penn state site", apply the Polaris Flex theme
        - Command: hax site site:theme --theme "polaris-flex-theme"

        For content generation requests:
        - Identify if content creation is implied (e.g., "create a site about..." or "about [topic]")
        - Suggest appropriate page titles
        - Generate three paragraphs of relevant content for each page in <p> tags
        - If "with video" is specified in a creation request, append <!-- VIDEO_PLACEHOLDER -->
        - For update requests, do not generate content—just specify the page title

        Respond with *only* a JSON object:
        {
            "workflow_type": "new_site" or "add_page" or "add_page_with_content" or "add_child_page_with_content" or "add_customized_child_page" or "update_page_with_video",
            "site_name": "optional_site_name",
            "pages": [
                {
                    "title": "page title",
                    "content": "HTML formatted content with three paragraphs",  # For creation workflows
                    "parent": "parent page title",
                    "customize_from_parent": true/false,
                    "customization_instruction": "how to adjust the parent content"
                }
            ],
            "confidence": 0.0 to 1.0,
            "special_instructions": ["any special notes"]
        }

        Rules for content generation:
        - Three paragraphs, each in <p> tags (for creation workflows)
        - Informative and factual
        - Use proper HTML formatting
        - Include <!-- VIDEO_PLACEHOLDER --> if "with video" is requested in creation workflows
        - For "update_page_with_video", only provide the page title in "pages"
        """

    def parse_query_with_llm(self, query: str, engine: str) -> Dict:
        if engine == "Claude":
            try:
                message = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    temperature=0.7,
                    system=self.create_system_prompt(),
                    messages=[{"role": "user", "content": f"Parse this request and generate complete response with titles and content if needed: {query}"}]
                )
                return self._parse_json_response(message.content[0].text)
            except Exception as e:
                error_msg = f"Error processing query with LLM: {str(e)}"
                if console.is_terminal:
                    console.print(f"[red]{error_msg}")
                raise ValueError(error_msg)
        elif engine == "ICDS":
            try:
                promptNew = f"Parse this request and generate complete response with titles and content if needed: {query}"
                llmAnswer = ollama.invoke(promptNew)
                return json.loads(llmAnswer)
            except Exception as e:
                error_msg = f"Error processing query with LLM: {str(e)}"
                if console.is_terminal:
                    console.print(f"[red]{error_msg}")
                raise ValueError(error_msg)

    def _parse_json_response(self, response_text: str) -> Dict:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError as e:
                    console.print(f"[red]Failed to extract valid JSON from response: {response_text}")
                    raise e
            else:
                console.print(f"[red]No JSON found in response: {response_text}")
                raise ValueError("Response does not contain valid JSON")

    def customize_content(self, parent_content: str, instruction: str) -> str:
        prompt = f"""Given the following content from a parent page:

        {parent_content}

        Customize this content based on the following instruction: {instruction}.
        Return *only* the customized content in HTML format with exactly three paragraphs, each wrapped in <p> tags.
        Do not include any introductory text, headers, or additional sentences beyond the three paragraphs.
        Ensure the content remains informative, factual, and relevant to the new title."""
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=2000,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            console.print(f"[red]Error customizing content with Claude: {str(e)}")
            raise ValueError(f"Failed to customize content: {str(e)}")

    def get_parent_content(self, parent_title: str, site_dir: str) -> str:
        site_json_path = os.path.join(site_dir, "site.json")
        if not os.path.exists(site_json_path):
            raise ValueError(f"site.json not found in {site_dir}")
        with open(site_json_path, 'r') as f:
            site_data = json.load(f)
        parent_location = next((item.get("location") for item in site_data.get("items", []) if item.get("title") == parent_title), None)
        if not parent_location:
            raise ValueError(f"Parent page '{parent_title}' not found in site.json")
        parent_file_path = os.path.join(site_dir, parent_location)
        if not os.path.exists(parent_file_path):
            raise ValueError(f"HTML file for parent '{parent_title}' not found at {parent_file_path}")
        with open(parent_file_path, 'r') as f:
            html_content = f.read()
        return html_content if html_content else "<p>No content available for this page.</p>"

    def get_page_location(self, page_title: str, site_dir: str) -> str:
        site_json_path = os.path.join(site_dir, "site.json")
        if not os.path.exists(site_json_path):
            raise ValueError(f"site.json not found in {site_dir}")
        with open(site_json_path, 'r') as f:
            site_data = json.load(f)
        page_location = next((item.get("location") for item in site_data.get("items", []) if item.get("title") == page_title), None)
        if not page_location:
            raise ValueError(f"Page '{page_title}' not found in site.json")
        return page_location

    def summarize_content(self, content: str) -> str:
        """Summarize HTML content into a concise query for YouTube search."""
        # Strip HTML tags
        soup = BeautifulSoup(content, 'html.parser')
        text = soup.get_text(separator=" ").strip()
        if not text:
            return "general educational content"  # Fallback if no content

        # Use LLM to summarize into a short query
        prompt = f"""Given the following text, summarize it into a concise search query (max 5-10 words) for finding relevant educational YouTube videos. Focus on key topics or themes:

        {text}

        Return *only* the summarized query, no extra text."""
        try:
            message = self.client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=50,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text.strip()
        except Exception as e:
            console.print(f"[red]Error summarizing content: {str(e)}")
            # Fallback: extract key phrases manually
            words = text.split()
            key_phrases = " ".join([w for w in words if len(w) > 4][:5])  # Simple heuristic
            return key_phrases or "general educational content"

    def validate_names(self, site_name: Optional[str], page_title: Optional[str]) -> None:
        if site_name and not re.match(r'^[a-zA-Z0-9_]+$', site_name):
            raise ValueError("Invalid site name. Use only letters, numbers, and underscores.")
        if page_title and not re.match(r'^[a-zA-Z0-9_\s\?\,\-]+$', page_title):
            raise ValueError(f"Invalid page title '{page_title}'. Use only letters, numbers, spaces, underscores, hyphens, commas, and question marks.")

def parse_duration(duration_str):
    match = re.match(r'PT(\d+H)?(\d+M)?(\d+S)?', duration_str)
    if not match:
        return 0
    hours = int(match.group(1)[:-1]) if match.group(1) else 0
    minutes = int(match.group(2)[:-1]) if match.group(2) else 0
    seconds = int(match.group(3)[:-1]) if match.group(3) else 0
    return hours * 3600 + minutes * 60 + seconds

def format_duration(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{seconds:02d}" if hours > 0 else f"{minutes}:{seconds:02d}"

def get_top_educational_videos(topic: str, max_results: int = 3, min_views: int = 1000) -> List[Dict]:
    if not Google_API_KEY:
        raise HTTPException(status_code=500, detail="Google API Key not configured")
    youtube = build("youtube", "v3", developerKey=Google_API_KEY)
    search_response = youtube.search().list(
        q=f"{topic} tutorial explain concept",
        part="id,snippet",
        maxResults=20,
        type="video",
        videoDefinition="high",
        relevanceLanguage="en"
    ).execute()
    video_ids = [item["id"]["videoId"] for item in search_response["items"]]
    channel_ids = list(set([item["snippet"]["channelId"] for item in search_response["items"]]))
    channels_response = youtube.channels().list(part="statistics,snippet", id=",".join(channel_ids)).execute()
    channel_info = {item["id"]: {"subscriber_count": int(item["statistics"].get("subscriberCount", 0)), "video_count": int(item["statistics"].get("videoCount", 0)), "channel_title": item["snippet"]["title"]} for item in channels_response["items"]}
    video_response = youtube.videos().list(part="snippet,statistics,contentDetails", id=",".join(video_ids)).execute()
    comments_data = {}
    for video_id in video_ids:
        try:
            comments_response = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=10, order="relevance").execute()
            comments_data[video_id] = comments_response.get("items", [])
        except:
            comments_data[video_id] = []
    videos = []
    educational_keywords = ["learn", "tutorial", "explained", "understand", "guide", "how to", "concept", "lesson", "course", "education"]
    for item in video_response["items"]:
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        video_id = item["id"]
        channel_id = item["snippet"]["channelId"]
        url = f"https://www.youtube.com/watch?v={video_id}"
        like_count = int(item["statistics"].get("likeCount", 0))
        view_count = int(item["statistics"].get("viewCount", 0))
        comment_count = int(item["statistics"].get("commentCount", 0))
        if view_count < min_views:
            continue
        like_view_ratio = (like_count / view_count) if view_count > 0 else 0
        engagement_ratio = ((like_count + comment_count) / view_count) if view_count > 0 else 0
        duration_seconds = parse_duration(item["contentDetails"]["duration"])
        publish_date = datetime.strptime(item["snippet"]["publishedAt"], "%Y-%m-%dT%H:%M:%SZ")
        days_since_publish = (datetime.now() - publish_date).days
        recency_score = max(0, 1 - (days_since_publish / 365))
        educational_score = sum(0.2 if keyword.lower() in title.lower() else 0 + 0.1 if keyword.lower() in description.lower() else 0 for keyword in educational_keywords)
        duration_score = 1 if 300 <= duration_seconds <= 1200 else duration_seconds / 300 if duration_seconds < 300 else max(0, 1 - ((duration_seconds - 1200) / 1800))
        channel_data = channel_info.get(channel_id, {})
        channel_authority = min(1, (channel_data.get("subscriber_count", 0) / 1000000) * 0.5 + (min(channel_data.get("video_count", 0), 100) / 100) * 0.5)
        comment_quality_score = 0
        if video_id in comments_data and comments_data[video_id]:
            comment_texts = [c["snippet"]["topLevelComment"]["snippet"]["textDisplay"].lower() for c in comments_data[video_id]]
            edu_keyword_count = sum(sum(1 for signal in ["understand", "helped", "learned", "clear", "thank", "great explanation"] if signal in comment) for comment in comment_texts)
            comment_quality_score = min(1, edu_keyword_count / (len(comment_texts) * 2)) if comment_texts else 0
        composite_score = (like_view_ratio * 0.3 + engagement_ratio * 0.15 + educational_score * 0.2 + duration_score * 0.1 + channel_authority * 0.15 + comment_quality_score * 0.05 + recency_score * 0.05)
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
    return sorted(videos, key=lambda x: x["composite_score"], reverse=True)[:max_results]

@app.post("/api/hax-cli", response_model=WorkflowResponse)
async def ask_ai_hax_cli(request: HaxCliRequest):
    workflow = HaxSiteWorkflow()

    try:
        parsed = workflow.parse_query_with_llm(request.query, request.engine)
        if parsed["confidence"] < 0.7:
            raise HTTPException(status_code=400, detail="Query unclear. Please rephrase your request.")
        
        workflow.validate_names(parsed.get("site_name"), None)
        
        workflow_steps = []
        workflow_type = parsed["workflow_type"]
        site_name = parsed.get("site_name")
        base_dir = os.path.dirname(os.path.abspath(__file__))
        site_base_dir = os.path.join(base_dir, "site")
        site_dirs = [d for d in os.listdir(site_base_dir) if os.path.isdir(os.path.join(site_base_dir, d))]
        if not site_dirs or len(site_dirs) > 1:
            raise ValueError("Expected exactly one site directory in 'site' folder")
        site_dir = os.path.join(site_base_dir, site_dirs[0])

        if workflow_type == "new_site":
            if not site_name:
                raise HTTPException(status_code=400, detail="Site name required for new site creation")
            workflow_steps.append({
                "description": "Create new Hax site",
                "command": f"hax site start --name {site_name} --y"
            })

        if "penn state site" in request.query.lower():
            workflow_steps.append({
                "description": "Apply Penn State theme",
                "command": 'hax site site:theme --theme "polaris-flex-theme"'
            })

        if workflow_type in ["add_page", "add_page_with_content", "add_child_page_with_content", "add_customized_child_page"]:
            for page in parsed.get("pages", []):
                title = page.get("title")
                content = page.get("content")
                parent = page.get("parent")
                customize_from_parent = page.get("customize_from_parent", False)
                customization_instruction = page.get("customization_instruction")
                workflow.validate_names(None, title)

                if "with video" in request.query.lower() and "about" in request.query.lower():
                    # Wait for AI content, then summarize it for video search
                    if content:
                        video_topic = workflow.summarize_content(content)
                        console.print(f"[yellow]Summarized content for video search: {video_topic}")
                    else:
                        video_topic = title if not parent else f"{parent} {title}"  # Fallback
                    videos = get_top_educational_videos(video_topic, max_results=2)
                    video_content = "".join(
                        f'<video-player source="{v["url"]}" title="{v["title"]}"></video-player>'
                        for v in videos
                    )
                    if content and "<!-- VIDEO_PLACEHOLDER -->" in content:
                        content = content.replace("<!-- VIDEO_PLACEHOLDER -->", video_content)
                    else:
                        content = (content or "") + video_content

                if workflow_type == "add_customized_child_page" and customize_from_parent and parent:
                    parent_content = workflow.get_parent_content(parent, site_dir)
                    content = workflow.customize_content(parent_content, customization_instruction)
                
                if content:
                    escaped_content = content.replace('"', '\\"')
                    workflow_steps.append({
                        "description": f'Add new page "{title}" with content',
                        "command": f'hax site node:add --title "{title}" --content "{escaped_content}" --y'
                    })
                else:
                    workflow_steps.append({
                        "description": f'Add new page "{title}"',
                        "command": f'hax site node:add --title "{title}" --y'
                    })

                if parent and workflow_type in ["add_child_page_with_content", "add_customized_child_page"]:
                    workflow.validate_names(None, parent)
                    workflow_steps.extend([
                        {"description": f'Wait 2 seconds for site.json to update with new page "{title}"', "command": 'sleep 2'},
                        {"description": f'Find parent page "{parent}" ID in site.json', "command": f'parentId=$(jq -r \'first(.items[] | select(.title == "{parent}") | .id)\' site.json) || echo "Parent \'{parent}\' not found, will create it" >&2'},
                        {"description": f'Create parent page "{parent}" if it doesn\'t exist', "command": f'[ -z "$parentId" ] && hax site node:add --title "{parent}" --y || echo "Parent \'{parent}\' already exists"'},
                        {"description": f'Wait 2 seconds for site.json to update with new parent page "{parent}"', "command": '[ -z "$parentId" ] && sleep 2 || true'},
                        {"description": f'Retry finding parent page "{parent}" ID in site.json after creation', "command": f'[ -z "$parentId" ] && parentId=$(jq -r \'first(.items[] | select(.title == "{parent}") | .id)\' site.json) || true'},
                        {"description": f'Find child page "{title}" ID in site.json', "command": f'childId=$(jq -r \'first(.items[] | select(.title == "{title}") | .id)\' site.json) || echo "Error: Child \'{title}\' not found" >&2'},
                        {"description": f'Associate child page "{title}" with parent "{parent}"', "command": '[ -n "$parentId" ] && [ -n "$childId" ] && hax site node:edit --item-id "$childId" --node-op parent --parent "$parentId" --y || echo "Skipping association due to missing IDs"'}
                    ])

        elif workflow_type == "update_page_with_video":
            for page in parsed.get("pages", []):
                title = page.get("title")
                workflow.validate_names(None, title)

                page_location = workflow.get_page_location(title, site_dir)
                page_file_path = os.path.join(site_dir, page_location)

                if not os.path.exists(page_file_path):
                    raise HTTPException(status_code=404, detail=f"Page file '{page_file_path}' not found")

                # Extract existing content and summarize for video search
                with open(page_file_path, 'r') as f:
                    existing_content = f.read()
                video_topic = workflow.summarize_content(existing_content)
                console.print(f"[yellow]Summarized existing content for video search: {video_topic}")

                videos = get_top_educational_videos(video_topic, max_results=2)
                video_content = "\n".join(
                    f'<video-player source="{v["url"]}" title="{v["title"]}"></video-player>'
                    for v in videos
                )

                with open(page_file_path, 'a') as f:
                    f.write(f"\n{video_content}")
                console.print(f"[green]Appended video content to {page_file_path}")

                workflow_steps.append({
                    "description": f'Append videos to existing page "{title}" at {page_location}',
                    "command": f"# Direct file append: added video content to {page_file_path}"
                })

        explanation = "Commands to execute:\n" + "\n".join(f"• {step['description']}: {step['command']}" for step in workflow_steps)
        if "special_instructions" in parsed:
            explanation += "\n\nSpecial notes:\n" + "\n".join(f"• {note}" for note in parsed["special_instructions"])
        
        commands_list = [step["command"] for step in workflow_steps if not step["command"].startswith("#")]
        if commands_list:
            script_path = os.path.join(site_dir, "aiHax.sh")
            recipe_content = "#!/bin/bash\n" + "\n".join(commands_list)
            is_new_file = not os.path.exists(script_path)
            
            try:
                with open(script_path, 'w') as f:
                    f.write(recipe_content)
                console.print(f"[green]Script file written successfully to {script_path}")
                if is_new_file:
                    os.chmod(script_path, 0o755)
                    console.print(f"[green]Execution bit set on {script_path}")
                result = subprocess.run(["bash", "aiHax.sh"], cwd=site_dir, check=True, capture_output=True, text=True)
                console.print(f"[green]Successfully executed {script_path} in {site_dir}")
                if result.stdout:
                    console.print(f"[yellow]Script output:\n{result.stdout}")
                if result.stderr:
                    console.print(f"[orange]Script warnings/errors:\n{result.stderr}")
            except subprocess.CalledProcessError as e:
                console.print(f"[red]Script execution failed: {e.stderr}")
                raise HTTPException(status_code=500, detail=f"Script execution failed: {e.stderr}")
            except Exception as e:
                console.print(f"[red]Error handling script file: {str(e)}")
                raise
        
        return WorkflowResponse(
            commands=[step["command"] for step in workflow_steps],
            descriptions=[step["description"] for step in workflow_steps],
            explanation=explanation,
            confidence=parsed["confidence"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/youtube-videos", response_model=YouTubeVideoResponse)
async def get_youtube_videos(request: YouTubeVideoRequest):
    try:
        videos = get_top_educational_videos(
            topic=request.topic,
            max_results=request.max_results,
            min_views=request.min_views
        )
        if not videos:
            raise HTTPException(status_code=404, detail=f"No videos found for '{request.topic}' with sufficient views.")
        return YouTubeVideoResponse(videos=videos, topic=request.topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching videos: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def get_html():
    html_path = pathlib.Path("index.html")
    if html_path.exists():
        return html_path.read_text()
    else:
        return """
        <html>
            <body>
                <h1>Error: HTML file not found</h1>
                <p>The index.html file was not found. Please make sure it exists in the same directory as the app.py file.</p>
            </body>
        </html>
        """

@app.get("/health")
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)