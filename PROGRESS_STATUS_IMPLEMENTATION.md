# Progress Status Implementation Guide

## 📊 Overview

Based on the analysis of Marker and your requirements, here are three implementable solutions for adding status information during file processing:

## 1. **Server-Sent Events (SSE) - Recommended** ✅

**Pros:**
- Simple to implement
- Works with existing HTTP infrastructure
- One-way communication (server to client)
- Automatic reconnection
- No additional dependencies

**Implementation Steps:**
1. Replace `subprocess.run()` with `subprocess.Popen()` to capture real-time output
2. Create SSE endpoint that streams progress updates
3. Parse Marker's debug output for progress indicators
4. Update frontend to consume SSE stream

## 2. **WebSocket Implementation**

**Pros:**
- Bi-directional communication
- Lower latency
- More control over connection

**Cons:**
- More complex setup
- Requires WebSocket library
- Connection management overhead

## 3. **Progress Estimation with Polling**

**Pros:**
- Works with current architecture
- No streaming required
- Simple to implement

**Cons:**
- Not real-time
- Estimation may be inaccurate
- Requires polling from frontend

## 🎯 Recommended Implementation: SSE

Here's what we can capture from Marker's output:

### Detectable Progress Stages:
1. **Model Loading** (5-15% progress)
   - "Loading detection model..."
   - "Loading OCR model..."
   - "Loading layout model..."

2. **Page Processing** (15-85% progress)
   - "Processing page X/Y"
   - "Running OCR on page X"
   - "Detecting layout for page X"

3. **LLM Enhancement** (85-95% progress)
   - "Sending to LLM..."
   - "Processing with Gemini/Ollama..."

4. **Finalization** (95-100% progress)
   - "Writing output files..."
   - "Extraction complete"

### Key Marker Output Patterns to Parse:
```python
# Marker debug output patterns
patterns = {
    'page_count': r'Found (\d+) pages',
    'current_page': r'Processing page (\d+)/(\d+)',
    'ocr_start': r'Running OCR',
    'layout_detection': r'Detecting layout',
    'llm_processing': r'LLM processing',
    'model_loading': r'Loading .* model',
    'completion': r'Finished processing'
}
```

## 📝 Implementation Plan

### Backend Changes:

1. **Add SSE endpoint** to `main.py`:
```python
from fastapi.responses import StreamingResponse
import asyncio

@app.post("/extract-progress")
async def extract_with_progress(...):
    async def progress_generator():
        # Yield SSE formatted progress updates
        yield f"data: {json.dumps({'status': 'starting'})}\n\n"
        # ... process with Popen and yield updates
    
    return StreamingResponse(
        progress_generator(),
        media_type="text/event-stream"
    )
```

2. **Process output parsing**:
```python
process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
for line in iter(process.stdout.readline, ''):
    # Parse line and yield appropriate progress update
```

### Frontend Changes:

1. **Add progress UI elements**:
- Progress bar
- Status message display
- Processing stage indicator
- Elapsed time counter

2. **EventSource connection**:
```javascript
const eventSource = new EventSource('/extract-progress');
eventSource.onmessage = (event) => {
    const data = JSON.parse(event.data);
    updateProgressUI(data);
};
```

## 🚀 Quick Win: Simple Status Messages

If you want a quick implementation without major changes:

1. Add status div to UI
2. Use Popen with line buffering
3. Send periodic fetch requests to check status
4. Display last known status message

## 📊 Expected User Experience

1. **Upload file** → "Initializing extraction..."
2. **Model loading** → "Loading AI models..." (5-15%)
3. **Page processing** → "Processing page 3 of 10..." (15-85%)
4. **LLM enhancement** → "Enhancing with AI..." (85-95%)
5. **Completion** → "Extraction complete!" (100%)

## 🔧 Performance Considerations

- SSE adds minimal overhead
- Progress updates every ~500ms is sufficient
- Parse only essential output lines
- Close SSE connection on completion/error

Would you like me to implement the SSE solution with real-time progress updates?
