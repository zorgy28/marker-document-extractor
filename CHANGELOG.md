# Changelog

All notable changes to the Marker Document Extractor project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] - 2025-01-25

### Fixed
- **Output Format Selection**: Resolved critical bug where selecting Markdown or HTML format would incorrectly return JSON content
  - The SSE endpoint now properly filters output files based on the selected format
  - Added format-specific file extension matching (.md for Markdown, .html for HTML, .json for JSON)
  - Implemented fallback logic to search for any output file if the expected format is not found
- **Save Configuration Button**: Restored functionality of the Save Configuration button that was missing from the UI
- **Debug Logging**: Enhanced logging throughout the application for better troubleshooting

### Changed
- Improved file selection logic in the SSE endpoint (main.py lines 640-675)
- Updated error handling with more descriptive messages
- Enhanced real-time progress tracking accuracy

### Technical Details
- File extension determination logic: main.py lines 646-651
- Fallback file search implementation: main.py lines 663-670
- Frontend displayResult function improvements for proper format handling

## [1.2.0] - 2025-01-24

### Added
- Server-Sent Events (SSE) support for real-time extraction progress
- GPU acceleration detection (MPS support for macOS)
- Enhanced debug mode with detailed logging options

### Fixed
- Image extraction and serving functionality
- User preferences persistence issues

### Changed
- Switched to single extraction mode with progress tracking by default
- Improved LLM model selection interface

## [1.1.0] - 2025-01-23

### Added
- GPU acceleration support using Metal Performance Shaders (MPS) on macOS
- Automatic detection of available hardware acceleration

### Fixed
- Save Configuration button not saving user preferences
- Output format selection not reflecting in the UI

### Changed
- Updated dependencies to latest stable versions
- Improved error messages for better user experience

## [1.0.0] - 2025-01-22

### Added
- Initial release of Marker Document Extractor
- Support for PDF to JSON, Markdown, and HTML conversion
- LLM processing integration with Google Gemini and Ollama
- Modern web interface with responsive design
- User preferences saving and loading
- Image extraction from documents
- Multi-language OCR support
- Advanced extraction options (page range, OCR control, etc.)
- Real-time processing status updates
