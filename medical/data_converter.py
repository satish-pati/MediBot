import pandas as pd
from langchain_core.documents import Document
import os
from pypdf import PdfReader
import gdown
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        return ""

def dataconverter():
    """Processes a PDF file from Google Drive and splits it into chunks for embedding."""
    
    docs = []
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1O-qQJ6PEcQ0y35At8yvUXntbCwn9Loug"  
    pdf_path = "./data/Medical_book.pdf"

    # Download the PDF from Google Drive
    gdown.download(f"https://drive.google.com/uc?id={file_id}", pdf_path, quiet=False)

    # Extract text from the downloaded PDF
    text_content = extract_text_from_pdf(pdf_path)

    if text_content:
        # Split the text into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=20,
            length_function=len,
        )
        chunks = text_splitter.split_text(text_content)

        # Create Document objects for each chunk
        for i, chunk in enumerate(chunks):
            metadata = {
                "file_name": os.path.basename(pdf_path),
                "chunk_number": i,
                "source": f"{os.path.basename(pdf_path)} - Chunk {i}",
            }
            doc = Document(page_content=chunk, metadata=metadata)
            docs.append(doc)

    return docs
