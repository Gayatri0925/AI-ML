import pdfplumber
from docx import Document
import io

def extract_text(file):
    """
    Extracts text from PDF, DOCX, or TXT resumes.
    Works with Streamlit UploadedFile object.
    """
    # Check type by file name
    filename = file.name.lower()
    
    if filename.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(file.read())) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    elif filename.endswith(".docx"):
        doc = Document(io.BytesIO(file.read()))
        text = "\n".join([p.text for p in doc.paragraphs])
        return text
    elif filename.endswith(".txt"):
        return file.getvalue().decode("utf-8")
    else:
        return ""
