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

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Hax CLI Workflow API",
    description="API for converting natural language queries into Hax CLI commands",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ICDS settings
client = QdrantClient(url="http://localhost:5903")
ollama = Ollama(base_url="http://localhost:5904", model="mistral")

# Rich console for better output formatting
console = Console()

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

class HaxSiteWorkflow:
    def __init__(self):
        """Initialize the Hax Site Workflow Agent with Anthropic client"""
        self.client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    def create_system_prompt(self):
        """Create a detailed system prompt for the Hax CLI workflow"""
        return """You are an AI assistant that interprets natural language queries into specific Hax CLI commands and generates content when requested. Assume commands are run from within the site directory, so no site name prefix is needed for paths.

        Available Workflows:
        1. Create a new website:
        - Command: hax site start --name [siteName] --y
        - Creates new directory with site
        - Opens localhost:3000

        2. Add new page(s) without content:
        - Command: hax site node:add --title [pageTitle] --y
        - Creates new page in the current site directory
        - Used when no specific content is requested

        3. Add new page(s) with content:
        - Command: hax site node:add --title [pageTitle] --content [htmlContent] --y
        - Creates new page with provided content in the current site directory
        - Used when the query implies content generation is needed

        4. Add a child page under a parent page with content:
        - Command: hax site node:add --title [childPageTitle] --content [htmlContent] --y
        - Followed by: hax site node:edit --item-id [childId] --node-op parent --parent [parentId] --y
        - Creates a new child page and associates it with the specified parent page
        - Used when the query specifies a child-parent relationship (e.g., "add a child page called X about Y under the parent page called Z")

        5. Add a customized child page:
        - If the query implies customization (e.g., "create a new page called 'black kittens from india' under 'black kittens'"),
          assume the child page title extends the parent page title and requires customized content based on the parent.
        - The parent page content will be provided separately, and you should indicate where to insert it in your response.

        Special Themes:
        - Penn State Theme: When request includes "penn state site", apply the Polaris Flex theme
        - Command: hax site site:theme --theme "polaris-flex-theme"

        For content generation requests:
        - Identify if the request implies content creation (e.g., "create a site about..." or "about Y")
        - For content requests, first suggest appropriate page titles
        - Then generate three paragraphs of relevant content for each page, unless customization is requested

        Respond with *only* a JSON object in the following format, with no additional text, explanations, or comments outside the JSON:
        {
            "workflow_type": "new_site" or "add_page" or "add_page_with_content" or "add_child_page_with_content" or "add_customized_child_page",
            "site_name": "optional_site_name",  # Only include if explicitly mentioned (e.g., for new site creation)
            "pages": [
                {
                    "title": "page title",
                    "content": "HTML formatted content with three paragraphs", # For add_page_with_content or add_child_page_with_content
                    "parent": "parent page title", # For add_child_page_with_content or add_customized_child_page
                    "customize_from_parent": true/false, # True for add_customized_child_page
                    "customization_instruction": "how to adjust the parent content" # For add_customized_child_page
                }
            ],
            "confidence": 0.0 to 1.0,
            "special_instructions": ["any special notes"]
        }

        For customized child page requests (e.g., "create a new page called 'black kittens from india' under 'black kittens'"):
        - Set "workflow_type" to "add_customized_child_page"
        - Set "customize_from_parent" to true
        - Provide "customization_instruction" based on the difference (e.g., "Adjust the content to focus on black kittens from India")
        - Do not generate content here; it will be customized later using the parent page content

        Rules for content generation:
        - Each page must have exactly three paragraphs
        - Each paragraph should be wrapped in <p> tags
        - Content should be informative and factual
        - Use proper HTML formatting
        - Make content specific to the page title and topic

        Examples (return only the JSON, nothing else):
        - "create a new penn state site called portfolio" → {"workflow_type": "new_site", "site_name": "portfolio", "pages": [], "special_instructions": ["Applying Penn State theme"], "confidence": 0.95}
        - "add an about page" → {"workflow_type": "add_page", "pages": [{"title": "about"}], "confidence": 0.90}
        - "add a contact page about new york" → {"workflow_type": "add_page_with_content", "pages": [{"title": "Contact", "content": "<p>...</p><p>...</p><p>...</p>"}], "confidence": 0.92}
        - "add a child page called 'Breeds' about dog breeds under the parent page called 'Dogs'" → {"workflow_type": "add_child_page_with_content", "pages": [{"title": "Breeds", "content": "<p>...</p><p>...</p><p>...</p>", "parent": "Dogs"}], "confidence": 0.94}
        - "create a new page called 'black kittens from india' under 'black kittens'" → {"workflow_type": "add_customized_child_page", "pages": [{"title": "black kittens from india", "parent": "black kittens", "customize_from_parent": true, "customization_instruction": "Adjust the content to focus on black kittens from India"}], "confidence": 0.95}
        """

    def parse_query_with_llm(self, query: str, engine: str) -> Dict:
        if engine == "Claude":
            try:
                message = self.client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=2000,
                    temperature=0.7,
                    system=self.create_system_prompt(),
                    messages=[
                        {
                            "role": "user",
                            "content": f"Parse this request and generate complete response with titles and content if needed: {query}"
                        }
                    ]
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
        """Helper method to parse JSON responses, with fallback to extract JSON if extra text is present"""
        try:
            # First attempt to parse the response directly as JSON
            return json.loads(response_text)
        except json.JSONDecodeError:
            # If direct parsing fails, try to extract JSON from the text using regex
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
        """Ask Claude to customize the parent page content based on the instruction"""
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
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            return message.content[0].text
        except Exception as e:
            console.print(f"[red]Error customizing content with Claude: {str(e)}")
            raise ValueError(f"Failed to customize content: {str(e)}")

    def get_parent_content(self, parent_title: str, site_dir: str) -> str:
        """Retrieve the content of the parent page from its HTML file referenced in site.json"""
        site_json_path = os.path.join(site_dir, "site.json")
        if not os.path.exists(site_json_path):
            raise ValueError(f"site.json not found in {site_dir}")
        
        with open(site_json_path, 'r') as f:
            site_data = json.load(f)
        
        parent_location = None
        for item in site_data.get("items", []):
            if item.get("title") == parent_title:
                parent_location = item.get("location")
                break
        
        if not parent_location:
            raise ValueError(f"Parent page '{parent_title}' not found in site.json")
        
        parent_file_path = os.path.join(site_dir, parent_location)
        if not os.path.exists(parent_file_path):
            raise ValueError(f"HTML file for parent '{parent_title}' not found at {parent_file_path}")
        
        with open(parent_file_path, 'r') as f:
            html_content = f.read()
        
        return html_content if html_content else "<p>No content available for this page.</p>"

    def validate_names(self, site_name: Optional[str], page_title: Optional[str]) -> None:
        if site_name and not re.match(r'^[a-zA-Z0-9_]+$', site_name):
            raise ValueError("Invalid site name. Use only letters, numbers, and underscores.")
        
        if page_title and not re.match(r'^[a-zA-Z0-9_\s\?\,\-]+$', page_title):
            raise ValueError(f"Invalid page title '{page_title}'. Use only letters, numbers, spaces, underscores, hyphens, commas, and question marks.")

@app.post("/api/hax-cli", response_model=WorkflowResponse)
async def ask_ai_hax_cli(request: HaxCliRequest):
    workflow = HaxSiteWorkflow()

    try:
        parsed = workflow.parse_query_with_llm(request.query, request.engine)
        
        if parsed["confidence"] < 0.7:
            raise HTTPException(
                status_code=400, 
                detail="Query unclear. Please rephrase your request."
            )
    
        workflow.validate_names(
            parsed.get("site_name"),
            None
        )
        
        workflow_steps = []
        workflow_type = parsed["workflow_type"]
        site_name = parsed.get("site_name")  # Only used for new site creation

        # Determine the site directory dynamically
        base_dir = os.path.dirname(os.path.abspath(__file__))
        site_base_dir = os.path.join(base_dir, "site")
        site_dirs = [d for d in os.listdir(site_base_dir) if os.path.isdir(os.path.join(site_base_dir, d))]
        if not site_dirs or len(site_dirs) > 1:
            raise ValueError("Expected exactly one site directory in 'site' folder")
        site_dir = os.path.join(site_base_dir, site_dirs[0])

        if workflow_type == "new_site":
            if not site_name:
                raise HTTPException(status_code=400, detail="Site name required for new site creation")
            workflow_steps = [
                {
                    "description": "Create new Hax site",
                    "command": f"hax site start --name {site_name} --y"
                }
            ]

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

                # Handle customized child page
                if workflow_type == "add_customized_child_page" and customize_from_parent and parent:
                    # Get parent content from its HTML file
                    parent_content = workflow.get_parent_content(parent, site_dir)
                    # Customize it with Claude
                    content = workflow.customize_content(parent_content, customization_instruction)
                
                # Step 1: Create the page (child or regular)
                if content:
                    workflow_steps.append({
                        "description": f'Add new page "{title}" with content',
                        "command": f'hax site node:add '
                                f'--title "{title}" '
                                f'--content "{content}" --y'
                    })
                else:
                    workflow_steps.append({
                        "description": f'Add new page "{title}"',
                        "command": f'hax site node:add '
                                f'--title "{title}" --y'
                    })

                # Step 2: If it's a child page (customized or not), add logic to find or create parent and associate
                if parent and workflow_type in ["add_child_page_with_content", "add_customized_child_page"]:
                    workflow.validate_names(None, parent)
                    
                    # Add a delay after child page creation
                    workflow_steps.append({
                        "description": f'Wait 2 seconds for site.json to update with new page "{title}"',
                        "command": 'sleep 2'
                    })
                    
                    # Try to find the parent page ID
                    workflow_steps.append({
                        "description": f'Find parent page "{parent}" ID in site.json',
                        "command": f'parentId=$(jq -r \'first(.items[] | select(.title == "{parent}") | .id)\' site.json) || echo "Parent \'{parent}\' not found, will create it" >&2'
                    })
                    
                    # If parentId is empty, create the parent page
                    workflow_steps.append({
                        "description": f'Create parent page "{parent}" if it doesn\'t exist',
                        "command": f'[ -z "$parentId" ] && hax site node:add --title "{parent}" --y || echo "Parent \'{parent}\' already exists"'
                    })
                    
                    # Add a delay after creating the parent page
                    workflow_steps.append({
                        "description": f'Wait 2 seconds for site.json to update with new parent page "{parent}"',
                        "command": '[ -z "$parentId" ] && sleep 2 || true'
                    })
                    
                    # Retry finding the parent page ID after creation
                    workflow_steps.append({
                        "description": f'Retry finding parent page "{parent}" ID in site.json after creation',
                        "command": f'[ -z "$parentId" ] && parentId=$(jq -r \'first(.items[] | select(.title == "{parent}") | .id)\' site.json) || true'
                    })
                    
                    # Find the child page ID
                    workflow_steps.append({
                        "description": f'Find child page "{title}" ID in site.json',
                        "command": f'childId=$(jq -r \'first(.items[] | select(.title == "{title}") | .id)\' site.json) || echo "Error: Child \'{title}\' not found" >&2'
                    })
                    
                    # Step 3: Associate child with parent using the retrieved IDs with error checking
                    workflow_steps.append({
                        "description": f'Associate child page "{title}" with parent "{parent}"',
                        "command": '[ -n "$parentId" ] && [ -n "$childId" ] && hax site node:edit --item-id "$childId" --node-op parent --parent "$parentId" --y || echo "Skipping association due to missing IDs"'
                    })

        explanation = "Commands to execute:\n" + "\n".join(
            f"• {step['description']}: {step['command']}" 
            for step in workflow_steps
        )
        
        if "special_instructions" in parsed:
            explanation += "\n\nSpecial notes:\n" + "\n".join(
                f"• {note}" for note in parsed["special_instructions"]
            )
        
        # Get the commands
        commands_list = [step["command"] for step in workflow_steps]
        
        # Determine the site directory dynamically
        script_path = os.path.join(site_dir, "aiHax.sh")
        
        # Write commands to aiHax.sh file with shebang
        recipe_content = "#!/bin/bash\n" + "\n".join(commands_list)
        
        # Check if the file is new
        is_new_file = not os.path.exists(script_path)
        
        try:
            with open(script_path, 'w') as f:
                f.write(recipe_content)
            console.print(f"[green]Script file written successfully to {script_path}")
            
            # Set execution bit if the file is new
            if is_new_file:
                os.chmod(script_path, 0o755)  # rwxr-xr-x
                console.print(f"[green]Execution bit set on {script_path}")
            
            # Change to the site directory and execute the script
            subprocess.run(["bash", "aiHax.sh"], cwd=site_dir, check=True)
            console.print(f"[green]Successfully executed {script_path} in {site_dir}")
            
        except Exception as e:
            console.print(f"[red]Error handling script file: {str(e)}")
            raise
        
        return WorkflowResponse(
            commands=commands_list,
            descriptions=[step["description"] for step in workflow_steps],
            explanation=explanation,
            confidence=parsed["confidence"]
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint to serve the HTML interface directly
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

# Simple health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# For running the application locally
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)