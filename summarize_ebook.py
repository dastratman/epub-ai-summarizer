import os
import re
import time
import json
import argparse
import html2text
import threading
import signal
import markdown
from bs4 import BeautifulSoup
import google.generativeai as genai
from ebooklib import epub
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
from pathlib import Path
import hashlib
import markdown
from markdown.extensions.extra import ExtraExtension
from markdown.extensions.nl2br import Nl2BrExtension
from markdown.extensions.sane_lists import SaneListExtension

# Add markdown library to requirements.txt
# pip install markdown

# Global termination flag
should_terminate = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ebook_summary.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def termination_checker():
    """Thread function to check for termination signals."""
    global should_terminate
    while not should_terminate:
        try:
            time.sleep(0.5)  # Check every half second
        except KeyboardInterrupt:
            should_terminate = True
            logger.info("Termination signal received. Cleaning up...")
            break


# Start termination checker thread
termination_thread = threading.Thread(target=termination_checker, daemon=True)
termination_thread.start()


def set_termination_handler():
    """Set a termination handler for the main thread."""
    def handler(signum, frame):
        global should_terminate
        should_terminate = True
        logger.info(f"Received signal {signum}. Initiating shutdown...")

    # Set the handler for SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


class ChapterSummarizer:
    """Class to handle ebook chapter summarization using Gemini."""

    def __init__(self, api_key, model_name='gemini-1.5-flash', cache_dir=".summary_cache",
                 summary_style="detailed", max_workers=3, retry_limit=3):
        """
        Initialize the summarizer.

        Args:
            api_key: Gemini API key
            model_name: Gemini model to use
            cache_dir: Directory to cache summaries
            summary_style: Style of summary (simple, concise, detailed, analytical)
            max_workers: Maximum number of parallel workers
            retry_limit: Number of API call retries
        """
        self.api_key = api_key
        self.model_name = model_name
        self.model = self._setup_gemini()
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.summary_style = summary_style
        self.max_workers = max_workers
        self.retry_limit = retry_limit

        # CSS for the output ebook
        self.css = """
            body {
                font-family: serif;
                line-height: 1.5;
                margin: 1em;
                padding: 0;
            }
            
            h1 {
                font-size: 1.5em;
                margin-bottom: 1em;
            }
            
            h2, h3, h4 {
                margin-top: 1.5em;
                margin-bottom: 0.8em;
            }
            
            p {
                margin-top: 1em;
                margin-bottom: 1em;
                text-align: justify;
            }
            
            strong, b {
                font-weight: bold;
            }
            
            em, i {
                font-style: italic;
            }
            
            
            /* Ensure spacing between sections */
            hr {
                margin: 2em 0;
                border: none;
                border-top: 1px solid #ccc;
            }
        """

    def _setup_gemini(self):
        """Configure the Gemini API with the key."""
        genai.configure(api_key=self.api_key)
        return genai.GenerativeModel(self.model_name)

    def _get_cache_filename(self, chapter_title, chapter_content):
        """Generate a cache filename based on chapter content hash."""
        # Create a unique hash based on the chapter title and content
        content_hash = hashlib.md5(
            (chapter_title + chapter_content[:5000]).encode()).hexdigest()
        safe_title = re.sub(r'[^\w\-_\.]', '_', chapter_title)[:50]
        return self.cache_dir / f"{safe_title}_{content_hash}.json"

    def _cache_summary(self, chapter_title, chapter_content, summary):
        """Cache a summary to avoid re-processing."""
        cache_file = self._get_cache_filename(chapter_title, chapter_content)
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump({
                'title': chapter_title,
                'timestamp': time.time(),
                'summary': summary
            }, f, ensure_ascii=False, indent=2)

    def _get_cached_summary(self, chapter_title, chapter_content):
        """Try to retrieve a cached summary."""
        cache_file = self._get_cache_filename(chapter_title, chapter_content)
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('summary')
            except Exception as e:
                logger.warning(f"Cache read error for {chapter_title}: {e}")
        return None

    def get_chapter_summary(self, chapter_title, chapter_text):
        """Get a summary of the chapter from Gemini with caching and retries."""
        global should_terminate

        # Check for termination flag
        if should_terminate:
            logger.info(
                f"Skipping summary for '{chapter_title}' due to termination request")
            return "Summary generation skipped due to termination request."

        # Check cache first
        cached = self._get_cached_summary(chapter_title, chapter_text)
        if cached:
            logger.info(f"Using cached summary for '{chapter_title}'")
            return cached

        # Prepare the prompt based on summary style
        prompt_templates = {
            "simple": """
                Please provide a comprehensive summary of the following section titled "{title}".
                
                Focus on capturing the essential information, main points, and significant details.
                                
                CONTENT:
                {text}
            """,
            "concise": """
                Provide a concise summary of the following section titled "{title}". Focus only on key 
                points and critical information. Keep it brief and to the point.
                                
                CONTENT:
                {text}
            """,
            "detailed": """
                Create a comprehensive summary of the following section titled "{title}". Include:
                
                1. Main points or developments
                2. Important details, facts, or events
                3. Significant concepts or arguments
                4. Key conclusions or outcomes
                
                Format your response with clear sections and maintain sufficient detail 
                to understand the content and its significance.
                                
                CONTENT:
                {text}
            """,
            "analytical": """
                Provide an analytical summary of the following section titled "{title}" with:
                
                1. OVERVIEW: Brief summary of the main content
                2. KEY POINTS: Analysis of the most important elements
                3. CONTEXT & CONNECTIONS: How this content relates to broader topics or themes
                4. SIGNIFICANCE: Why this content matters and its implications
                
                This should be detailed and insightful, focusing on deeper meaning and significance.
                                
                CONTENT:
                {text}
            """
        }

        template = prompt_templates.get(
            self.summary_style, prompt_templates["detailed"])
        prompt = template.format(
            title=chapter_title, text=chapter_text[:30000])

        # Try multiple times in case of API errors
        for attempt in range(self.retry_limit):
            # Check for termination flag
            if should_terminate:
                logger.info(
                    f"Cancelling summary generation for '{chapter_title}'")
                return "Summary generation cancelled due to termination request."

            try:
                response = self.model.generate_content(prompt)
                summary = response.text

                # Cache the successful result
                self._cache_summary(chapter_title, chapter_text, summary)
                return summary

            except Exception as e:
                logger.warning(
                    f"Attempt {attempt+1}/{self.retry_limit} failed for '{chapter_title}': {e}")
                if attempt < self.retry_limit - 1 and not should_terminate:
                    # Exponential backoff
                    time.sleep(2 ** attempt)
                else:
                    logger.error(
                        f"Failed to summarize '{chapter_title}' after {self.retry_limit} attempts")
                    return f"*Summary generation failed after {self.retry_limit} attempts.*"

    def extract_text_from_html(self, html_content):
        """Extract clean text from HTML content with better formatting preservation."""
        # Use BeautifulSoup to clean the HTML first
        soup = BeautifulSoup(html_content, 'html.parser')

        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()

        # Configure html2text for better formatting
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_tables = False
        h.body_width = 0  # Don't wrap text
        text = h.handle(str(soup))

        # Clean up the text
        text = re.sub(r'\n{3,}', '\n\n', text)  # Replace multiple newlines
        text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove markdown links
        text = text.strip()

        return text

    def process_epub(self, input_path, output_path):
        """Process the EPUB file and create a new one with summaries."""
        global should_terminate

        try:
            # Read the input EPUB
            book = epub.read_epub(input_path)

            # Create a new EPUB for summaries
            summary_book = epub.EpubBook()

            # Get title, safely
            title_metadata = book.get_metadata('DC', 'title')
            orig_title = title_metadata[0][0] if title_metadata and title_metadata[0] else "Unknown Title"
            summary_book.set_title(f"Summary of {orig_title}")

            # Add minimal required metadata
            summary_book.add_metadata('DC', 'language', 'en')
            summary_book.add_metadata(
                'DC', 'identifier', f'summary_{int(time.time())}')
            summary_book.add_author("Generated by Gemini AI")

            # Safely copy selected metadata from original book
            try:
                for metadata_type in ['creator', 'subject', 'description', 'publisher', 'source']:
                    meta_values = book.get_metadata('DC', metadata_type)
                    if meta_values:
                        for meta_value in meta_values:
                            # Check that value isn't None
                            if meta_value and meta_value[0]:
                                summary_book.add_metadata(
                                    'DC', metadata_type, meta_value[0])
            except Exception as e:
                logger.warning(f"Error copying metadata: {e}")

            # Add CSS
            css_file = epub.EpubItem(
                uid="style_default",
                file_name="style/default.css",
                media_type="text/css",
                content=self.css
            )
            summary_book.add_item(css_file)

            # Create introduction chapter
            intro = epub.EpubHtml(title="Introduction",
                                  file_name="intro.xhtml")
            intro.add_item(css_file)
            intro.content = f"""<html>
            <head><link rel="stylesheet" href="style/default.css" type="text/css" /></head>
            <body>
                <h1>Introduction</h1>
                <p>This book contains AI-generated summaries of each section from 
                "{orig_title}".</p>
                <p>Summaries were created using Google's Gemini AI model ({self.model_name})
                with the '{self.summary_style}' style.</p>
                <p>Generated on: {time.strftime('%Y-%m-%d')}</p>
            </body>
            </html>"""
            summary_book.add_item(intro)

            # Keep track of our chapters for the table of contents
            chapters = [intro]
            toc = [(epub.Section('Summaries'), [])]

            # Identify real chapters (filtering out front/back matter)
            content_items = []
            for item_id, linear in book.spine:
                if should_terminate:
                    break

                if item_id == 'nav':
                    continue

                item = book.get_item_with_id(item_id)

                if isinstance(item, epub.EpubHtml):
                    # Extract text to check if this is likely a content chapter
                    soup = BeautifulSoup(item.content, 'html.parser')
                    text_content = soup.get_text()

                    # Heuristic: If it has enough text, it's likely a content chapter
                    if len(text_content) > 500:
                        title_tag = soup.find(['h1', 'h2', 'h3'])
                        title = title_tag.text.strip(
                        ) if title_tag else f"Section {len(content_items) + 1}"
                        content_items.append((item, title))

            logger.info(
                f"Found {len(content_items)} content sections to summarize")

            # Function to process a single chapter
            def process_chapter(idx, item_tuple):
                global should_terminate

                if should_terminate:
                    return idx, None, item_tuple[1], "Skipped due to termination request"

                item, chapter_title = item_tuple
                try:
                    # Extract text
                    chapter_text = self.extract_text_from_html(item.content)

                    # Get summary
                    logger.info(f"Processing: {chapter_title}")
                    summary = self.get_chapter_summary(
                        chapter_title, chapter_text)

                    # Format the summary as HTML with nicer structure
                    formatted_summary = self._format_summary_html(
                        summary, chapter_title)

                    # Create the chapter
                    summary_chapter = epub.EpubHtml(
                        title=f"Summary: {chapter_title}",
                        file_name=f"summary_{idx:03d}.xhtml"
                    )
                    summary_chapter.add_item(css_file)
                    summary_chapter.content = formatted_summary

                    return idx, summary_chapter, chapter_title, None

                except Exception as e:
                    error_msg = str(e)
                    logger.error(
                        f"Error processing '{chapter_title}': {error_msg}", exc_info=True)
                    return idx, None, chapter_title, error_msg

            # Process chapters in parallel with a progress bar
            processed_chapters = []

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(process_chapter, i, item_tuple): i
                           for i, item_tuple in enumerate(content_items)}

                with tqdm(total=len(content_items), desc="Summarizing sections") as progress:
                    for future in futures:
                        if should_terminate:
                            executor.shutdown(wait=False)
                            logger.info(
                                "Cancelling remaining tasks due to termination request")
                            break

                        idx, chapter, title, error = future.result()
                        if chapter:
                            processed_chapters.append((idx, chapter, title))
                        elif error:
                            logger.error(
                                f"Failed to process section {title}: {error}")
                        progress.update(1)

                        # Check for termination flag periodically
                        if should_terminate:
                            break

            # If termination was requested, save what we have so far
            if should_terminate:
                logger.info("Termination requested. Saving partial results...")

            # Sort by original order and add to book
            processed_chapters.sort(key=lambda x: x[0])
            for idx, chapter, title in processed_chapters:
                if chapter:
                    summary_book.add_item(chapter)
                    chapters.append(chapter)
                    toc[0][1].append(chapter)

            # Add default NCX and Nav files
            summary_book.add_item(epub.EpubNcx())
            nav = epub.EpubNav()
            nav.add_item(css_file)
            summary_book.add_item(nav)

            # Define the book's spine
            summary_book.spine = ['nav'] + chapters

            # Add TOC
            summary_book.toc = toc

            # Write the epub file
            if len(chapters) > 1:  # Only write if we have at least some content
                epub.write_epub(output_path, summary_book)
                logger.info(f"Wrote summary EPUB to {output_path}")
                return len(processed_chapters)
            else:
                logger.error(
                    "No sections were successfully processed. EPUB not created.")
                return 0

        except Exception as e:
            logger.error(f"Error in process_epub: {e}", exc_info=True)
            raise

    def _format_summary_html(self, summary, chapter_title):
        """Format the summary text as nicely structured HTML with proper list structure."""
        # Start with basic HTML structure
        formatted_html = "<html>\n<head><link rel=\"stylesheet\" href=\"style/default.css\" type=\"text/css\" /></head>\n<body>\n"
        formatted_html += f"<h1>Summary: {chapter_title}</h1>\n"

        # Fix common markdown formatting issues before converting
        cleaned_summary = self._clean_markdown_lists(summary)

        # Convert to HTML
        html_content = markdown.markdown(
            cleaned_summary,
            extensions=[ExtraExtension(), Nl2BrExtension(),
                        SaneListExtension()],
            output_format='html5'
        )

        # Add the converted HTML content
        formatted_html += html_content
        formatted_html += "\n</body>\n</html>"

        return formatted_html

    def _clean_markdown_lists(self, text):
        """Fix markdown list formatting issues for proper HTML conversion."""
        # Split into lines for processing
        lines = text.split('\n')
        result_lines = []

        # Track if we're in a list to add proper spacing
        in_list = False
        list_indent_level = 0

        for i, line in enumerate(lines):
            # Check if this is a list item (numbered or bullet)
            list_match = re.match(r'^(\s*)(\*|\d+\.)\s+(.*)', line)

            if list_match:
                spaces, marker, content = list_match.groups()
                indent_level = len(spaces) // 4

                # Format as proper markdown list item
                if marker == '*':
                    # Bullet list
                    formatted_line = ('    ' * indent_level) + '* ' + content
                else:
                    # Numbered list
                    formatted_line = ('    ' * indent_level) + \
                        marker + ' ' + content

                # Add empty line before list starts if needed
                if not in_list and i > 0 and result_lines and result_lines[-1].strip():
                    result_lines.append('')

                result_lines.append(formatted_line)
                in_list = True
                list_indent_level = indent_level
            else:
                # Not a list item - check if we're exiting a list
                if in_list and line.strip():
                    # Add empty line after list ends
                    if result_lines and result_lines[-1].strip():
                        result_lines.append('')
                    in_list = False

                result_lines.append(line)

        return '\n'.join(result_lines)

    def _preprocess_markdown(self, text):
        """Pre-process text to ensure proper markdown formatting."""
        # Fix numbered lists (ensure proper spacing)
        text = re.sub(r'(\n\d+\.)\s+', r'\n\1 ', text)

        # Fix bullet lists with asterisks (ensure proper spacing)
        text = re.sub(r'(\n\*)\s+', r'\n\1 ', text)

        # Fix markdown lists that start with dashes
        text = re.sub(r'(\n-)\s+', r'\n\1 ', text)

        # Ensure empty line before lists for proper markdown parsing
        text = re.sub(r'([^\n])\n(\d+\.\s)', r'\1\n\n\2', text)
        text = re.sub(r'([^\n])\n(\*\s)', r'\1\n\n\2', text)
        text = re.sub(r'([^\n])\n(-\s)', r'\1\n\n\2', text)

        # Convert markdown lists that use incorrect format (1. text<br> style)
        lines = text.split('\n')
        in_list = False
        for i in range(len(lines)):
            # Check for list items in paragraphs using <br>
            if not in_list and (re.match(r'^\d+\.\s+', lines[i]) or re.match(r'^\*\s+', lines[i]) or re.match(r'^-\s+', lines[i])):
                # Insert blank line before list starts
                if i > 0 and lines[i-1].strip():
                    lines[i] = '\n' + lines[i]
                in_list = True
            elif in_list and not (re.match(r'^\d+\.\s+', lines[i]) or re.match(r'^\*\s+', lines[i]) or re.match(r'^-\s+', lines[i])):
                # List ended
                if lines[i].strip():
                    lines[i] = '\n' + lines[i]
                in_list = False

        return '\n'.join(lines)


def main():
    """Main function to parse arguments and run the processing."""
    global should_terminate

    # Set termination handler
    set_termination_handler()

    parser = argparse.ArgumentParser(
        description="Create chapter summaries from an EPUB using Gemini AI")
    parser.add_argument("input_epub", help="Path to the input EPUB file")
    parser.add_argument(
        "--output", help="Path for the output EPUB (default: input_summaries.epub)")
    parser.add_argument(
        "--api-key", help="Gemini API key. Alternatively, set GEMINI_API_KEY environment variable.")
    parser.add_argument("--model", default="gemini-1.5-flash",
                        help="Gemini model to use (default: gemini-1.5-flash)")
    parser.add_argument("--style", default="detailed",
                        choices=["simple", "concise",
                                 "detailed", "analytical"],
                        help="Style of summary (default: detailed)")
    parser.add_argument("--max-workers", type=int, default=3,
                        help="Maximum number of parallel processes (default: 3)")
    parser.add_argument("--cache-dir", default=".summary_cache",
                        help="Directory to cache summaries (default: .summary_cache)")
    parser.add_argument("--retry-limit", type=int, default=3,
                        help="Number of API call retries (default: 3)")

    args = parser.parse_args()

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base_name = os.path.splitext(args.input_epub)[0]
        output_path = f"{base_name}_summaries.epub"

    # Get API key
    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Provide it with --api-key or set GEMINI_API_KEY environment variable.")

    # Create the summarizer
    summarizer = ChapterSummarizer(
        api_key=api_key,
        model_name=args.model,
        cache_dir=args.cache_dir,
        summary_style=args.style,
        max_workers=args.max_workers,
        retry_limit=args.retry_limit
    )

    try:
        # Process the EPUB
        logger.info(
            f"Processing {args.input_epub} with {args.style} summary style...")
        chapter_count = summarizer.process_epub(args.input_epub, output_path)

        if should_terminate:
            logger.info("Processing was terminated early.")
            print("Processing was terminated early. Partial results were saved.")
        elif chapter_count > 0:
            logger.info(
                f"Done! Created summaries for {chapter_count} sections.")
            logger.info(f"Output saved to: {output_path}")
            print(f"Done! Created summaries for {chapter_count} sections.")
            print(f"Output saved to: {output_path}")
        else:
            logger.error("No sections were successfully processed.")
            print("Error: No sections were successfully processed.")
            return 1

    except Exception as e:
        logger.error(f"Error processing ebook: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
