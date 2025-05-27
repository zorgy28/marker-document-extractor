from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import os
import subprocess
import json
import requests
from dotenv import load_dotenv
import torch  # Add torch import for MPS detection
import asyncio
from typing import AsyncGenerator
import re
import uuid
import shutil
from typing import Optional
import time

# Load environment variables from .env file
load_dotenv()

# Set Ollama environment variables for better performance
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "2h")  # Default keep_alive time
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")  # Limit memory usage

# Check for MPS availability at startup
if torch.backends.mps.is_available():
    print(" GPU Acceleration: MPS (Metal Performance Shaders) is available!")
    print(f"   PyTorch version: {torch.__version__}")
    print("   Marker will use your M3 Max GPU for faster processing")
else:
    print(" MPS not available - Marker will use CPU")

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
    image_patterns = ['*.jpg', '*.jpeg', '*.png', '*.gif', '.bmp', '*.webp']
    
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

        # Create output directory
        output_dir = os.path.join(tmp_dir, "output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare marker command with proper arguments
        marker_cmd = [
            "marker",
            tmp_dir,  # Input folder (not individual file)
            "--output_dir", output_dir,  # Output directory (different from input)
            "--output_format", outputFormat.lower(),
            "--workers", str(workers),
            "--debug"  # Enable debug mode for progress tracking
        ]
        
        # Add page range if specified
        if pageRange:
            marker_cmd.extend(["--page_range", pageRange])
        
        # Add image extraction flag
        if not extractImages:
            marker_cmd.append("--disable_image_extraction")
        
        # Add language override
        if languages:
            marker_cmd.extend(["--languages", languages])
        
        # Add force OCR option
        if forceOcr:
            marker_cmd.extend(["--force_ocr"])
        
        # Add strip existing OCR option  
        if stripExistingOcr:
            marker_cmd.extend(["--strip_existing_ocr"])
        
        # Add LLM service configuration if specified
        if useLlm:
            if llmService == "openai":
                marker_cmd.extend(["--llm_service", "marker.services.openai.OpenAIService"])
            elif llmService == "anthropic":
                marker_cmd.extend(["--llm_service", "marker.services.anthropic.AnthropicService"])
            elif llmService == "ollama":
                marker_cmd.extend(["--llm_service", "marker.services.ollama.OllamaService"])
                
                # Add Ollama-specific configuration
                if ollamaModel:
                    marker_cmd.extend(["--ollama_model", ollamaModel])
                
                if ollamaBaseUrl:
                    marker_cmd.extend(["--ollama_base_url", ollamaBaseUrl])
        
        print(f"[SSE] Marker command output format: --output_format {outputFormat.lower()}")
        print(f"Running Marker CLI: {' '.join(marker_cmd)}")
        
        # Set timeout for LLM processing (longer timeout for Ollama)
        if useLlm and llmService == "ollama":
            timeout = 600  # 10 minutes for Ollama (model loading + processing)
        elif useLlm:
            timeout = 300  # 5 minutes for cloud LLMs
        else:
            timeout = 120  # 2 minutes for regular processing
        
        try:
            result = subprocess.run(marker_cmd, capture_output=True, text=True, timeout=timeout)
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
            # Try to get more specific error information
            error_msg = "Marker conversion failed"
            
            # Check for common dependency errors
            stderr_output = ""
            if result.stderr:
                stderr_output = result.stderr
                if "weasyprint" in stderr_output.lower():
                    error_msg = "Missing dependency: weasyprint is required for DOCX conversion. Run: pip install weasyprint"
                elif "mammoth" in stderr_output.lower():
                    error_msg = "Missing dependency: mammoth is required for DOCX conversion. Run: pip install mammoth"
                elif "openpyxl" in stderr_output.lower():
                    error_msg = "Missing dependency: openpyxl is required for Excel conversion. Run: pip install openpyxl"
                elif "python-pptx" in stderr_output.lower():
                    error_msg = "Missing dependency: python-pptx is required for PowerPoint conversion. Run: pip install python-pptx"
                elif "ebooklib" in stderr_output.lower():
                    error_msg = "Missing dependency: ebooklib is required for EPUB conversion. Run: pip install ebooklib"
                elif stderr_output:
                    # Include part of the actual error
                    error_lines = stderr_output.strip().split('\n')
                    if error_lines:
                        error_msg = f"Marker error: {error_lines[-1][:200]}"
            
            return JSONResponse(
                status_code=500,
                content={"error": error_msg}
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

class Extractor:
    async def _send_heartbeats(self):
        """Send periodic heartbeat messages to keep connection alive"""
        while True:
            await asyncio.sleep(5)  # Send heartbeat every 5 seconds
            yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

    async def extract_with_progress(
        self,
        file: UploadFile = File(...),
        outputFormat: str = Form("json"),
        useLlm: bool = Form(False),
        llmService: str = Form("gemini"),
        googleApiKey: Optional[str] = Form(None),
        ollamaBaseUrl: str = Form("http://localhost:11434"),
        ollamaModel: str = Form("llama2"),
        extractImages: bool = Form(True),
        pageRange: Optional[str] = Form(None),
        languages: Optional[str] = Form(None),
        workers: int = Form(1),
        forceOcr: bool = Form(False),
        stripExistingOcr: bool = Form(False),
        paginateOutput: bool = Form(False),
        debugMode: bool = Form(False)
    ):
        """Extract document with real-time progress updates using SSE"""
        
        # Read file content before starting the async generator
        file_content = await file.read()
        input_filename = file.filename
        
        async def generate_progress():
            temp_dir = None
            session_id = str(int(time.time() * 1000))
            session_images_dir = os.path.join("extracted_images", session_id)
            
            try:
                # Log the received parameters
                print(f"[SSE] Received outputFormat: {outputFormat}")
                print(f"[SSE] Debug mode: {debugMode}")
                
                # Create temporary directory
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, input_filename)
                
                with open(file_path, "wb") as f:
                    f.write(file_content)
                
                print(f"[SSE] Starting extraction for file: {input_filename}")
                
                yield f"data: {json.dumps({'type': 'start', 'status': 'initializing', 'message': 'Starting extraction...', 'session_id': session_id})}\n\n"
                await asyncio.sleep(0)
                
                # Load LLM model if needed
                if llmService and not useLlm:
                    load_status = await load_ollama_model(llmService)
                    yield f"data: {json.dumps({'type': 'progress', 'status': 'model_loaded', 'message': f'Loaded {llmService} model', 'progress': 10})}\n\n"
                    await asyncio.sleep(0)
                
                # Create output directory
                output_dir = os.path.join(temp_dir, "output")
                os.makedirs(output_dir, exist_ok=True)
                
                # Prepare marker command with proper arguments
                marker_cmd = [
                    temp_dir,  # Input folder (not individual file)
                    "--output_dir", output_dir,  # Output directory (different from input)
                    "--output_format", outputFormat.lower(),
                    "--workers", str(workers),
                    "--debug"  # Enable debug mode for progress tracking
                ]
                
                # Add page range if specified
                if pageRange:
                    marker_cmd.extend(["--page_range", pageRange])
                
                # Add image extraction flag
                if not extractImages:
                    marker_cmd.append("--disable_image_extraction")
                
                # Add language override
                if languages:
                    marker_cmd.extend(["--languages", languages])
                
                # Add force OCR option
                if forceOcr:
                    marker_cmd.extend(["--force_ocr"])
                
                # Add strip existing OCR option  
                if stripExistingOcr:
                    marker_cmd.extend(["--strip_existing_ocr"])
                
                # Add LLM service configuration if specified
                if useLlm:
                    if llmService == "gemini":
                        marker_cmd.extend(["--llm_service", "marker.services.google.GoogleService"])
                        if googleApiKey:
                            env["GOOGLE_API_KEY"] = googleApiKey
                    elif llmService == "ollama":
                        marker_cmd.extend(["--llm_service", "marker.services.ollama.OllamaService"])
                        
                        # Add Ollama-specific configuration
                        if ollamaModel:
                            marker_cmd.extend(["--ollama_model", ollamaModel])
                        
                        if ollamaBaseUrl:
                            marker_cmd.extend(["--ollama_base_url", ollamaBaseUrl])
                
                print(f"[SSE] Marker command output format: --output_format {outputFormat.lower()}")
                # Remove all single and double quotes from marker_cmd arguments
                # Ensure marker_cmd is constructed with plain strings only
                print(f"[SSE] marker_cmd list (PLAIN): {marker_cmd}")
                # Check for empty or None arguments
                for idx, arg in enumerate(marker_cmd):
                    if arg is None or (isinstance(arg, str) and arg.strip() == ""):
                        error_msg = f"Invalid argument in marker_cmd at position {idx}: {repr(arg)}\nFull marker_cmd: {[repr(a) for a in marker_cmd]}"
                        print(f"[SSE] {error_msg}")
                        yield f"data: {json.dumps({'type': 'error', 'status': 'failed', 'message': error_msg})}\n\n"
                        return
                print(f"[SSE] Running command: {' '.join(marker_cmd)}")
                # Validate input and output directories are not the same
                if os.path.abspath(temp_dir) == os.path.abspath(output_dir):
                    error_msg = f"Input and output directories are the same! This will cause marker CLI to fail.\nInput: {temp_dir}\nOutput: {output_dir}"
                    print(f"[SSE] {error_msg}")
                    yield f"data: {json.dumps({'type': 'error', 'status': 'failed', 'message': error_msg})}\n\n"
                    return
                
                # Set timeout for LLM processing (longer timeout for Ollama)
                if useLlm and llmService == "ollama":
                    timeout = 600  # 10 minutes for Ollama (model loading + processing)
                elif useLlm:
                    timeout = 300  # 5 minutes for cloud LLMs
                else:
                    timeout = 120  # 2 minutes for regular processing
                
                # Execute Marker command
                process = await asyncio.create_subprocess_exec(
                    'marker',
                    *marker_cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'}
                )
                
                yield f"data: {json.dumps({'type': 'progress', 'status': 'processing', 'message': 'Starting document processing...', 'progress': 15})}\n\n"
                await asyncio.sleep(0)
                
                # Progress tracking variables
                total_pages = None
                current_page = 0
                
                async def read_stream_generator(stream, stream_name):
                    nonlocal total_pages, current_page
                    buffer = ""
                    
                    while True:
                        chunk = await stream.read(1024)
                        if not chunk:
                            break
                        
                        text = chunk.decode('utf-8', errors='ignore')
                        buffer += text
                        lines = buffer.split('\n')
                        buffer = lines[-1]  # Keep incomplete line in buffer
                        
                        for line in lines[:-1]:
                            line = line.strip()
                            if not line:
                                continue
                            
                            print(f"[SSE] {stream_name}: {line}")  # Debug
                            
                            # Always send debug logs first if debug mode is enabled
                            if debugMode:
                                yield f"data: {json.dumps({'type': 'debug', 'message': f'[{stream_name}] {line}'})}\n\n"
                            
                            # Update progress based on output
                            if "Loading" in line and "model" in line:
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'loading_models', 'message': 'Loading AI models...', 'progress': 12})}\n\n"
                            
                            elif "Converting" in line:
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'converting', 'message': 'Converting document...', 'progress': 10})}\n\n"
                            
                            elif "Dumped" in line and ("layout" in line or "PDF" in line or "document" in line):
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'analyzing', 'message': 'Analyzing document structure...', 'progress': 30})}\n\n"
                            
                            elif re.search(r'Processing page (\d+)', line):
                                match = re.search(r'Processing page (\d+)', line)
                                if match:
                                    current_page = int(match.group(1))
                                    if total_pages:
                                        progress = 30 + int((current_page / total_pages) * 55)
                                        yield f"data: {json.dumps({'type': 'progress', 'status': 'processing_page', 'message': f'Processing page {current_page} of {total_pages}', 'progress': progress, 'current_page': current_page, 'total_pages': total_pages})}\n\n"
                            
                            elif "Page" in line and "/" in line:
                                # Extract page numbers  
                                page_match = re.search(r'Page (\d+)/(\d+)', line)
                                if page_match:
                                    current_page = int(page_match.group(1))
                                    total_pages = int(page_match.group(2))
                                    
                                    # Calculate progress (30-85%)
                                    page_progress = 30 + int((current_page / total_pages) * 55)
                                    yield f"data: {json.dumps({'type': 'progress', 'status': 'processing', 'message': f'Processing page {current_page} of {total_pages}', 'progress': page_progress, 'current_page': current_page, 'total_pages': total_pages})}\n\n"
                            
                            elif "OCR" in line:
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'ocr', 'message': 'Running OCR on images...', 'progress': 20})}\n\n"
                            
                            elif "Layout" in line or "detection" in line:
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'layout_detection', 'message': 'Analyzing document layout...', 'progress': 25})}\n\n"
                            
                            elif "LLM" in line or "Gemini" in line or "Ollama" in line:
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'llm_processing', 'message': 'Enhancing with AI...', 'progress': 87})}\n\n"
                            
                            elif "Writing" in line or "Saving" in line:
                                yield f"data: {json.dumps({'type': 'progress', 'status': 'saving', 'message': 'Saving results...', 'progress': 92})}\n\n"
                    
                    # Process any remaining data in buffer
                    if buffer.strip():
                        print(f"[SSE] {stream_name} (final): {buffer.strip()}")
                
                # Read streams concurrently
                async def monitor_process():
                    error_messages = []
                    captured_stderr = []
                    
                    async for output in read_stream_generator(process.stdout, "STDOUT"):
                        yield output
                    
                    async for output in read_stream_generator(process.stderr, "STDERR"):
                        # Capture error messages from stderr
                        if '"type": "debug"' in output:
                            try:
                                # Extract the actual message from debug output
                                data = json.loads(output.split("data: ")[1].strip())
                                if "[STDERR]" in data.get("message", ""):
                                    captured_stderr.append(data["message"].replace("[STDERR] ", ""))
                            except:
                                pass
                        error_messages.append(output)
                        yield output
                    
                    return_code = await process.wait()
                    print(f"[SSE] Process completed with return code: {return_code}")
                    
                    if return_code != 0:
                        error_msg = "Marker conversion failed"
                        
                        # Check for common dependency errors in captured stderr
                        stderr_text = "\n".join(captured_stderr)
                        
                        if "weasyprint" in stderr_text.lower():
                            error_msg = "Missing dependency: weasyprint is required for DOCX conversion. Please reinstall with: pip install marker-pdf[full]"
                        elif "mammoth" in stderr_text.lower():
                            error_msg = "Missing dependency: mammoth is required for DOCX conversion. Please reinstall with: pip install marker-pdf[full]"
                        elif "openpyxl" in stderr_text.lower():
                            error_msg = "Missing dependency: openpyxl is required for Excel conversion. Please reinstall with: pip install marker-pdf[full]"
                        elif "python-pptx" in stderr_text.lower():
                            error_msg = "Missing dependency: python-pptx is required for PowerPoint conversion. Please reinstall with: pip install marker-pdf[full]"
                        elif "ebooklib" in stderr_text.lower():
                            error_msg = "Missing dependency: ebooklib is required for EPUB conversion. Please reinstall with: pip install marker-pdf[full]"
                        elif captured_stderr:
                            # Include part of the actual error
                            if captured_stderr:
                                error_msg = f"Marker error: {captured_stderr[-1][:200]}"
                        
                        yield f"data: {json.dumps({'type': 'error', 'status': 'failed', 'message': error_msg})}\n\n"
                        yield {"return_code": return_code}  # Signal failure
                        return
                    
                    yield f"data: {json.dumps({'type': 'progress', 'status': 'finalizing', 'message': 'Processing output...', 'progress': 95})}\n\n"
                    await asyncio.sleep(0)
                    yield {"return_code": return_code}  # Signal success
                    
                # Run the process and monitor output
                process_return_code = None
                
                async for output in monitor_process():
                    if isinstance(output, dict) and "return_code" in output:
                        process_return_code = output["return_code"]
                    elif isinstance(output, str):
                        yield output
                
                # Only proceed if extraction was successful
                if process_return_code != 0:
                    return
                
                # Create session directory for images
                session_images_dir = os.path.join("extracted_images", session_id)
                os.makedirs(session_images_dir, exist_ok=True)
                
                # Find the output file - Marker saves it in subdirectories
                output_files = []
                print(f"[SSE] All files in temp directory: {os.listdir(temp_dir)}")
                print(f"[SSE] Looking for output format: {outputFormat}")
                
                # Determine the expected file extension based on format
                if outputFormat.lower() == 'markdown':
                    expected_extension = '.md'
                elif outputFormat.lower() == 'html':
                    expected_extension = '.html'
                else:
                    expected_extension = '.json'
                
                # Define source document extensions to exclude
                source_extensions = (
                    '.pdf',  # PDF
                    '.png', '.jpg', '.jpeg', '.jpe', '.jif', '.jfif', '.jfi',  # JPEG variants
                    '.gif', '.bmp', '.dib',  # Basic image formats
                    '.tiff', '.tif',  # TIFF variants
                    '.webp', '.ico', '.heic', '.heif',  # Modern image formats
                    '.pptx',  # PowerPoint
                    '.docx',  # Word
                    '.xlsx', '.xls',  # Excel
                    '.html', '.htm',  # HTML
                    '.epub'  # E-books
                )
                
                # Look for files with the expected extension
                output_dir = os.path.join(temp_dir, "output")
                for root, dirs, files in os.walk(output_dir):  # Search in output directory
                    for file in files:
                        file_path = os.path.join(root, file)
                        # Look for files with the expected extension
                        if file.endswith(expected_extension) and not file.endswith('config.json') and not file.lower().endswith(source_extensions):
                            output_files.append(file_path)
                            print(f"[SSE] Found output file: {file_path}")
                
                if not output_files:
                    # Fallback: look for any markdown, json, or html file
                    print(f"[SSE] No {expected_extension} files found, looking for any output file")
                    for root, dirs, files in os.walk(output_dir):  # Search in output directory
                        for file in files:
                            file_path = os.path.join(root, file)
                            if file.endswith(('.md', '.json', '.html')) and not file.endswith('config.json') and not file.lower().endswith(source_extensions):
                                output_files.append(file_path)
                                print(f"[SSE] Found fallback output file: {file_path}")
                
                if not output_files:
                    yield f"data: {json.dumps({'type': 'error', 'status': 'failed', 'message': 'No output file generated by Marker'})}\n\n"
                    return
                
                # Use the first output file found
                output_file = output_files[0]
                print(f"[SSE] Using output file: {output_file}")
                
                # Read the output file
                with open(output_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Update image paths to point to our served images
                content = update_image_paths(content, session_id)
                
                # Copy extracted images
                copy_extracted_images(temp_dir, session_images_dir)
                
                yield f"data: {json.dumps({'type': 'progress', 'status': 'complete', 'message': 'Extraction complete!', 'progress': 100})}\n\n"
                await asyncio.sleep(0)
                
                # Send final result
                yield f"data: {json.dumps({'type': 'result', 'status': 'success', 'data': {'content': content, 'format': outputFormat, 'session_id': session_id}, 'session_id': session_id})}\n\n"
                    
            except Exception as e:
                print(f"[SSE] Fatal error in generate_progress: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'status': 'failed', 'message': f'Server error: {str(e)}'})}\n\n"
        
        return StreamingResponse(
            generate_progress(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable proxy buffering
                "Content-Type": "text/event-stream"
            }
        )

extractor = Extractor()
app.post("/extract-progress")(extractor.extract_with_progress)
