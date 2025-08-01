import fitz  # PyMuPDF for PDFs
import docx
import email
from typing import Tuple, Optional

def parse_pdf(file_path: str) -> Tuple[str, dict]:
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    metadata = {"page_count": doc.page_count}
    return text, metadata

def parse_docx(file_path: str) -> Tuple[str, dict]:
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    metadata = {}
    return text, metadata

def parse_email(file_path: str) -> Tuple[str, dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        msg = email.message_from_file(f)
    text = ""
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                text += part.get_payload(decode=True).decode(errors="ignore")
    else:
        text = msg.get_payload(decode=True).decode(errors="ignore")
    metadata = {
        "from": msg.get("From"),
        "to": msg.get("To"),
        "subject": msg.get("Subject"),
        "date": msg.get("Date"),
    }
    return text, metadata