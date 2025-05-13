# EPUB AI Summarizer

![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

Automatically generate chapter-by-chapter summaries for any EPUB ebook using Google's Gemini AI, creating a new, well-formatted EPUB containing all summaries.

## Features

- **Complete Automation**: Process entire ebooks with a single command
- **Multiple Summary Styles**: Choose between concise, detailed, or analytical summaries
- **Smart Caching**: Avoid redundant API calls with built-in caching
- **Parallel Processing**: Summarize multiple chapters simultaneously
- **Beautiful Output**: Well-structured EPUB with styled formatting
- **Robust Error Handling**: Automatic retries and detailed logging

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/epub-ai-summarizer.git
   cd epub-ai-summarizer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Get a Google AI API key:
   - Visit the [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create an API key

## Usage

### Basic Usage

```bash
python summarize_ebook.py your_ebook.epub --api-key YOUR_API_KEY
```

### Environment Variable

You can also set your API key as an environment variable:

```bash
# Set API key in environment
export GEMINI_API_KEY=your_api_key

# Run without specifying key on command line
python summarize_ebook.py your_ebook.epub
```

### Advanced Options

```bash
python summarize_ebook.py your_ebook.epub \
    --style analytical \
    --max-workers 4 \
    --output custom_name.epub \
    --cache-dir .my_cache \
    --retry-limit 5
```

## Command Line Options

| Option          | Description                                                         |
| --------------- | ------------------------------------------------------------------- |
| `input_epub`    | Path to the input EPUB file                                         |
| `--output`      | Path for the output EPUB (default: input_summaries.epub)            |
| `--api-key`     | Gemini API key (or use GEMINI_API_KEY environment variable)         |
| `--model`       | Gemini model to use (default: gemini-pro)                           |
| `--style`       | Summary style: concise, detailed, or analytical (default: detailed) |
| `--max-workers` | Maximum number of parallel processes (default: 3)                   |
| `--cache-dir`   | Directory to cache summaries (default: .summary_cache)              |
| `--retry-limit` | Number of API call retries (default: 3)                             |

## Summary Styles

- **Concise**: Brief summaries focusing on key plot points only
- **Detailed**: Comprehensive summaries covering plot, characters, and themes
- **Analytical**: In-depth analysis including literary techniques and significance

## How It Works

1. **Chapter Extraction**: The script parses your EPUB file to identify and extract chapters
2. **Parallel Processing**: Multiple chapters are processed simultaneously for efficiency
3. **AI Summarization**: Each chapter is sent to Google's Gemini AI model for summarization
4. **Smart Caching**: Summaries are cached to avoid redundant API calls if the script is rerun
5. **Format & Compile**: Results are formatted with HTML/CSS and compiled into a new EPUB
6. **Error Handling**: Built-in retry logic and detailed logging ensure reliability

## Examples

### Generate Concise Summaries
```bash
python summarize_ebook.py my_novel.epub --style concise
```

### Generate Detailed Summaries with 5 Workers
```bash
python summarize_ebook.py my_novel.epub --style detailed --max-workers 5
```

### Generate Analytical Summaries
```bash
python summarize_ebook.py textbook.epub --style analytical
```

## Limitations

- API rate limits may apply depending on your Gemini account
- Very large chapters might be truncated due to token limits
- Chapter detection is based on common EPUB structures and may not work for all books

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.