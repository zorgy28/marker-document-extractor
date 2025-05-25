# Marker Document Extractor

A web-based document extraction tool that converts PDF documents to structured formats (JSON, Markdown, HTML) using the powerful [Marker](https://github.com/VikParuchuri/marker) library. The application features a modern web interface with support for LLM-enhanced processing via Google Gemini or local Ollama models.

## 🚀 Features

- **Multiple Output Formats**: Convert PDFs to JSON, Markdown, or HTML
- **LLM Processing**: Enhanced extraction using Google Gemini or local Ollama models
- **Image Extraction**: Automatically extract and serve images from documents
- **Advanced Options**: Page range selection, OCR control, multi-language support
- **User Preferences**: Save and load extraction preferences
- **Real-time Processing**: Live status updates during document processing
- **Modern UI**: Clean, responsive web interface

## 📋 Prerequisites

Before installing, ensure you have:

- Python 3.8 or higher
- [Marker](https://github.com/VikParuchuri/marker) library installed
- (Optional) [Ollama](https://ollama.ai/) for local LLM processing
- (Optional) Google API key for Gemini processing

## 🛠️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/marker-document-extractor.git
cd marker-document-extractor
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Marker

Follow the official [Marker installation guide](https://github.com/VikParuchuri/marker#installation):

```bash
pip install marker-pdf
```

For GPU acceleration (recommended):
```bash
pip install marker-pdf[torch]
```

### 5. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Google API Key for Marker LLM processing
# Get your API key from: https://makersuite.google.com/app/apikey
GOOGLE_API_KEY=your_google_api_key_here
```

### 6. (Optional) Install and Configure Ollama

For local LLM processing:

1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull a compatible model:
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   # or
   ollama pull devstral
   ```

## 🚀 Usage

### Starting the Application

```bash
python main.py
```

The application will start on `http://localhost:8000`

### Using the Web Interface

1. **Upload Document**: Select a PDF file to extract
2. **Choose Output Format**: JSON, Markdown, or HTML
3. **Configure LLM** (optional):
   - Select Google Gemini or Ollama
   - For Ollama: Ensure it's running and select a model
   - For Gemini: Provide your API key
4. **Set Advanced Options**:
   - Extract images
   - Page range selection
   - Language specification
   - Worker count for parallel processing
5. **Extract**: Click the Extract button to process your document

### API Endpoints

- `GET /` - Serve the web interface
- `POST /extract` - Extract document content
- `GET /images/{session_id}/{filename}` - Serve extracted images
- `GET /sessions` - List active sessions
- `DELETE /cleanup/{session_id}` - Clean up session data
- `GET /ollama/status` - Check Ollama status and models
- `GET /preferences` - Get saved preferences
- `POST /preferences` - Save user preferences

## 📁 Project Structure

```
marker-document-extractor/
├── main.py                 # FastAPI application
├── index.html             # Web interface
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables
├── user_preferences.json  # Saved user preferences
├── extracted_images/      # Served image directory
└── README.md             # This file
```

## ⚙️ Configuration

### Environment Variables

- `GOOGLE_API_KEY`: Google Gemini API key for LLM processing
- `OLLAMA_KEEP_ALIVE`: How long to keep Ollama models loaded (default: 2h)
- `OLLAMA_MAX_LOADED_MODELS`: Maximum number of loaded models (default: 1)

### User Preferences

The application saves user preferences automatically, including:
- LLM service selection
- Model preferences
- Output format
- Processing options

## 🔧 Advanced Usage

### Command Line Parameters

The application supports all Marker command-line parameters:

- `--output_format`: Output format (json, markdown, html)
- `--use_llm`: Enable LLM processing
- `--page_range`: Specific page range (e.g., "1-5")
- `--languages`: Language codes for OCR (e.g., "en,es")
- `--workers`: Number of parallel workers
- `--force_ocr`: Force OCR even for text-based PDFs
- `--strip_existing_ocr`: Remove existing OCR before processing
- `--paginate_output`: Paginate the output
- `--debug`: Enable debug mode

### Ollama Configuration

For optimal performance with Ollama:

1. Increase memory allocation:
   ```bash
   export OLLAMA_KEEP_ALIVE=2h
   export OLLAMA_MAX_LOADED_MODELS=1
   ```

2. Use recommended models:
   - `llama2`: General purpose
   - `mistral`: Fast and accurate
   - `devstral`: Code-optimized

## 🐛 Troubleshooting

### Common Issues

1. **Marker not found**: Ensure Marker is properly installed
   ```bash
   pip install marker-pdf
   ```

2. **Ollama connection failed**: Check if Ollama is running
   ```bash
   ollama serve
   ```

3. **Google API errors**: Verify your API key in `.env`

4. **Memory issues**: Reduce worker count or use smaller models

### Debug Mode

Enable debug mode for detailed logging:
- Check the "Debug mode" option in the web interface
- Or set `debugMode: true` in your preferences

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit your changes: `git commit -am 'Add feature'`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Marker](https://github.com/VikParuchuri/marker) - The core document extraction library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern web framework
- [Ollama](https://ollama.ai/) - Local LLM runtime

## 📞 Support

If you encounter any issues or have questions:

1. Check the [troubleshooting section](#-troubleshooting)
2. Review the [Marker documentation](https://github.com/VikParuchuri/marker)
3. Open an issue on GitHub

---

**Note**: This application requires the Marker library to be properly installed and configured. Please follow the [official Marker installation guide](https://github.com/VikParuchuri/marker#installation) for your specific system.
