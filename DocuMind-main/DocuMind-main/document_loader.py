
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader


def load_pdf(file_source):
    """
    Extract text from a PDF and preserve page numbers.

    file_source can be a file path or a Streamlit uploaded file.
    Returns a list of dictionaries containing page numbers and text.
    """
    if isinstance(file_source, (str, Path)):
        reader = PdfReader(str(file_source))
    else:
        reader = PdfReader(BytesIO(file_source.getvalue()))

    pages = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""

        if text.strip():
            pages.append({
                "page": page_number,
                "text": text.strip()
            })

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF. "
            "It may be scanned or image-only."
        )

    return pages
