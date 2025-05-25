from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import subprocess
import json
import shutil
import uuid
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Set Ollama environment variables for better performance
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "2h")  # Default keep_alive time
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")  # Limit memory usage

def update_image_paths(content: str, session_id: str) -> str:
    """Update image paths in HTML/Markdown to point to our served images"""
    import re
    
    # Pattern to match various image references from Marker:
    # _page_X_Picture_Y.ext, _page_X_Figure_Y.ext, or any _page_X_*.ext pattern
    image_pattern = r'(_page_\d+_[A-Za-z]+_\d+\.\w+)'
    
    def replace_path(match):
        image_name = match.group(1)
        return f"/images/{session_id}/{image_name}"
    
    return re.sub(image_pattern, replace_path, content)

def copy_extracted_images(source_dir: str, dest_dir: str):
    """Copy extracted images from temporary directory to served directory"""
    import glob
    
    # Find all image files in the source directory and subdirectories
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp', '*.webp']
    
    for root, dirs, files in os.walk(source_dir):
        for pattern in image_patterns:
            for image_file in glob.glob(os.path.join(root, pattern)):
                if os.path.isfile(image_file):
                    filename = os.path.basename(image_file)
                    dest_path = os.path.join(dest_dir, filename)
                    try:
                        shutil.copy2(image_file, dest_path)
                        print(f"Copied image: {filename} to {dest_path}")
                    except Exception as e:
                        print(f"Error copying image {filename}: {e}")

app = FastAPI()

# Create directories for serving extracted content
IMAGES_DIR = "extracted_images"
os.makedirs(IMAGES_DIR, exist_ok=True)

# Serve extracted images
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")

@app.get("/")
async def root():
    return FileResponse("index.html")

@app.post("/extract")
async def extract_document(
    file: UploadFile = File(...),
    outputFormat: str = Form("json"),
    useLlm: bool = Form(False),
    llmService: str = Form("gemini"),
    googleApiKey: str = Form(""),
    ollamaBaseUrl: str = Form("http://localhost:11434"),
    ollamaModel: str = Form("llama2"),
    extractImages: bool = Form(True),
    pageRange: str = Form(""),
    languages: str = Form(""),
    workers: int = Form(1),
    forceOcr: bool = Form(False),
    stripExistingOcr: bool = Form(False),
    paginateOutput: bool = Form(False),
    debugMode: bool = Form(False)
):
    print("/extract endpoint called")
    with tempfile.TemporaryDirectory() as tmp_dir:
        file_path = os.path.join(tmp_dir, file.filename)
        file_bytes = await file.read()
        print(f"Received file: {file.filename}, size: {len(file_bytes)} bytes")
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        # List contents of temp directory before running Marker
        print(f"Temp directory contents before Marker: {os.listdir(tmp_dir)}")

        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Build marker command with options
        cmd = ["marker", tmp_dir, "--output_dir", output_dir, "--output_format", outputFormat]
        
        if useLlm:
            cmd.append("--use_llm")
            
            if llmService == "ollama":
                # Configure Ollama (now supported in Marker 1.7.3+)
                cmd.extend(["--ollama_base_url", ollamaBaseUrl])
                cmd.extend(["--ollama_model", ollamaModel])
                print(f"Using Ollama: {ollamaBaseUrl} with model {ollamaModel}")
                
                # Pre-load the model and set keep_alive to keep it in memory longer
                try:
                    preload_payload = {
                        "model": ollamaModel,
                        "prompt": ".",  # Minimal prompt to load the model
                        "stream": False,
                        "keep_alive": "2h"  # Keep model loaded for 2 hours after use
                    }
                    requests.post(f"{ollamaBaseUrl}/api/generate", 
                                json=preload_payload, 
                                timeout=10)
                    print(f"Pre-loaded {ollamaModel} with 2-hour keep_alive")
                except Exception as e:
                    print(f"Warning: Could not pre-load model: {e}")
            else:
                # Use Gemini
                api_key = googleApiKey or os.getenv("GOOGLE_API_KEY")
                if api_key:
                    cmd.extend(["--google_api_key", api_key])
                    print(f"Using Google API key: {api_key[:10]}...")
                else:
                    print("Warning: LLM processing enabled but no API key provided")
        
        if not extractImages:
            cmd.append("--disable_image_extraction")
            
        if pageRange:
            cmd.extend(["--page_range", pageRange])
            
        if languages:
            cmd.extend(["--languages", languages])
            
        if workers > 1:
            cmd.extend(["--workers", str(workers)])
            
        if forceOcr:
            cmd.append("--force_ocr")
            
        if stripExistingOcr:
            cmd.append("--strip_existing_ocr")
            
        if paginateOutput:
            cmd.append("--paginate_output")
            
        if debugMode:
            cmd.append("--debug")
        
        print(f"Running Marker CLI: {' '.join(cmd)}")
        
        # Set timeout for LLM processing (longer timeout for Ollama)
        if useLlm and llmService == "ollama":
            timeout = 600  # 10 minutes for Ollama (model loading + processing)
        elif useLlm:
            timeout = 300  # 5 minutes for cloud LLMs
        else:
            timeout = 120  # 2 minutes for regular processing
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return JSONResponse(
                status_code=500,
                content={"error": f"Processing timed out after {timeout} seconds. Try reducing the document size or disabling LLM processing."}
            )
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to run Marker CLI: {str(e)}"}
            )
        print(f"Marker CLI exit code: {result.returncode}")
        if result.stdout:
            print(f"Marker CLI stdout: {result.stdout}")
        if result.stderr:
            print(f"Marker CLI stderr: {result.stderr}")

        if result.returncode != 0:
            return JSONResponse(
                status_code=500,
                content={"error": result.stderr}
            )

        # Generate unique session ID for this extraction
        session_id = str(uuid.uuid4())[:8]
        session_images_dir = os.path.join(IMAGES_DIR, session_id)
        os.makedirs(session_images_dir, exist_ok=True)
        
        # Find output files based on format
        extracted = None
        expected_extensions = {
            "json": ".json",
            "markdown": ".md", 
            "html": ".html"
        }
        
        file_extension = expected_extensions.get(outputFormat, ".json")
        
        try:
            print(f"Output directory exists: {os.path.exists(output_dir)}")
            if os.path.exists(output_dir):
                print(f"Output directory contents: {os.listdir(output_dir)}")
                for fname in os.listdir(output_dir):
                    print(f"Found file: {fname}")
                    if fname.endswith(file_extension):
                        file_path = os.path.join(output_dir, fname)
                        print(f"Reading output file: {file_path}")
                        with open(file_path, "r") as out_f:
                            if outputFormat == "json":
                                extracted = json.load(out_f)
                            else:
                                content = out_f.read()
                                # Update image paths to point to our served images
                                content = update_image_paths(content, session_id)
                                extracted = {"content": content, "format": outputFormat, "session_id": session_id}
                        break
            else:
                print(f"Output directory {output_dir} does not exist!")
        except Exception as e:
            print(f"Error reading output directory: {e}")
            
        # Also check if there are any files in the temp directory itself
        print(f"Temp directory contents after Marker: {os.listdir(tmp_dir)}")
        
        # Check for output files anywhere in temp directory recursively
        for root, dirs, files in os.walk(tmp_dir):
            for file in files:
                if file.endswith(file_extension):
                    file_path = os.path.join(root, file)
                    print(f"Found output file in subdirectory: {file_path}")
                    if extracted is None:
                        try:
                            with open(file_path, "r") as out_f:
                                if outputFormat == "json":
                                    extracted = json.load(out_f)
                                else:
                                    content = out_f.read()
                                    # Update image paths to point to our served images
                                    content = update_image_paths(content, session_id)
                                    extracted = {"content": content, "format": outputFormat, "session_id": session_id}
                            print(f"Successfully loaded output from: {file_path}")
                        except Exception as e:
                            print(f"Error reading output file {file_path}: {e}")

        # Copy any extracted images to our served directory
        copy_extracted_images(tmp_dir, session_images_dir)

        if extracted is None:
            return JSONResponse(
                status_code=500,
                content={"error": f"No output {outputFormat} file produced by Marker. Check server logs for details."}
            )

        if outputFormat == "json":
            print(f"Extraction result: {json.dumps(extracted)[:500]}")
        else:
            print(f"Extraction result: {extracted['content'][:500]}")
        
        # Save preferences after successful extraction
        try:
            preferences = {
                "llmService": llmService,
                "ollamaModel": ollamaModel,
                "ollamaBaseUrl": ollamaBaseUrl,
                "outputFormat": outputFormat,
                "extractImages": extractImages,
                "workers": workers,
                "useLlm": useLlm
            }
            with open("user_preferences.json", "w") as f:
                json.dump(preferences, f, indent=2)
            print("Preferences saved successfully")
        except Exception as e:
            print(f"Warning: Could not save preferences: {e}")
            
        return extracted

@app.delete("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up extracted images for a specific session"""
    session_dir = os.path.join(IMAGES_DIR, session_id)
    if os.path.exists(session_dir):
        try:
            shutil.rmtree(session_dir)
            return {"message": f"Session {session_id} cleaned up successfully"}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to cleanup session: {str(e)}"}
            )
    else:
        return {"message": "Session not found"}

@app.get("/sessions")
async def list_sessions():
    """List all active sessions"""
    sessions = []
    if os.path.exists(IMAGES_DIR):
        for item in os.listdir(IMAGES_DIR):
            session_path = os.path.join(IMAGES_DIR, item)
            if os.path.isdir(session_path):
                image_count = len([f for f in os.listdir(session_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'))])
                sessions.append({
                    "session_id": item,
                    "image_count": image_count
                })
    return {"sessions": sessions}

@app.get("/ollama/status")
async def check_ollama_status(base_url: str = "http://localhost:11434"):
    """Check if Ollama is running and list available models"""
    try:
        # Check if Ollama is running
        response = requests.get(f"{base_url}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [model.get("name", "").split(":")[0] for model in models]
            return {
                "status": "running",
                "base_url": base_url,
                "models": list(set(model_names))  # Remove duplicates
            }
        else:
            return {
                "status": "error",
                "message": f"Ollama responded with status {response.status_code}"
            }
    except requests.exceptions.RequestException as e:
        return {
            "status": "offline",
            "message": f"Cannot connect to Ollama at {base_url}: {str(e)}"
        }

@app.get("/preferences")
async def get_preferences():
    """Get saved user preferences"""
    preferences_file = "user_preferences.json"
    default_preferences = {
        "llmService": "ollama",  # Default to Ollama since it's available
        "ollamaModel": "llama3.3",  # Use the model you have loaded
        "ollamaBaseUrl": "http://localhost:11434",
        "outputFormat": "json",
        "extractImages": True,
        "workers": 1,
        "useLlm": True  # Enable LLM by default
    }
    
    try:
        if os.path.exists(preferences_file):
            with open(preferences_file, "r") as f:
                preferences = json.load(f)
                # Merge with defaults for any missing keys
                return {**default_preferences, **preferences}
        else:
            return default_preferences
    except Exception as e:
        print(f"Error loading preferences: {e}")
        return default_preferences

@app.post("/preferences")
async def save_preferences(preferences: dict):
    """Save user preferences"""
    preferences_file = "user_preferences.json"
    try:
        with open(preferences_file, "w") as f:
            json.dump(preferences, f, indent=2)
        return {"status": "success", "message": "Preferences saved"}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to save preferences: {str(e)}"}
        )
