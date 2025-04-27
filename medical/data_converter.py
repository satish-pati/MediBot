```python
# Import necessary libraries
import pandas as pd # Import pandas library for data manipulation
from langchain_core.documents import Document # Import Document class from langchain_core
import os # Import os library for file operations
from pypdf import PdfReader # Import PdfReader class from pypdf for reading PDF files
import gdown # Import gdown library for downloading files from Google Drive
from langchain.text_splitter import RecursiveCharacterTextSplitter # Import RecursiveCharacterTextSplitter class from langchain

# Extracts text from a PDF file and returns the extracted text as a string
def extract_text_from_pdf(pdf_path):
"""
Extracts text from a PDF file.
Args:
pdf_path (str): The path to the PDF file.
Returns:
str: The extracted text.
"""
try:
# Create a PDF reader object to read the PDF file
reader = PdfReader(pdf_path) # Initialize a PdfReader object

# Initialize an empty string to store the extracted text
text = "" # Initialize an empty string to store the extracted text

# Iterate over each page in the PDF
for page in reader.pages: # Loop through each page in the PDF
# Extract the text from the current page and append it to the text string
text += page.extract_text() + "\n" # Extract text from the current page and append to the text string

# Return the extracted text, stripping any leading or trailing whitespace
return text.strip() # Return the extracted text, removing leading/trailing whitespace
except Exception as e:
# If an error occurs, print an error message and return an empty string
print(f"Error extracting text from {pdf_path}: {e}") # Print an error message if extraction fails
return "" # Return an empty string if extraction fails

# Processes a PDF file from Google Drive, splits it into chunks, and creates Document objects for each chunk
def dataconverter():
"""
Processes a PDF file from Google Drive and splits it into chunks for embedding.
Returns:
list: A list of Document objects, each representing a chunk of the PDF text.
"""
# Initialize an empty list to store the Document objects
docs = [] # Initialize an empty list to store Document objects

# Google Drive file ID (replace with your actual file ID)
file_id = "1O-qQJ6PEcQ0y35At8yvUXntbCwn9Loug" # Specify the Google Drive file ID

# Local path to save the PDF file
pdf_path = "./data/Medical_book.pdf" # Specify the local path to save the PDF file

# Download the PDF from Google Drive using gdown library
gdown.download(f"https://drive.google.com/uc?id={file_id}", pdf_path, quiet=False) # Download the PDF from Google Drive

# Extract text from the downloaded PDF using the extract_text_from_pdf function
text_content = extract_text_from_pdf(pdf_path) # Extract text from the downloaded PDF

# If text was extracted, proceed with processing
if text_content: # Check if text was extracted successfully
# Split the text into smaller chunks using a recursive character text splitter
text_splitter = RecursiveCharacterTextSplitter(
chunk_size=500, # Specify the maximum size of each chunk
chunk_overlap=20, # Specify the overlap between chunks
length_function=len, # Specify the function to calculate chunk length
) # Initialize a RecursiveCharacterTextSplitter object
chunks = text_splitter.split_text(text_content) # Split the text into chunks

# Create Document objects for each chunk
for i, chunk in enumerate(chunks): # Loop through each chunk
# Define metadata for the current chunk
metadata = {
"file_name": os.path.basename(pdf_path), # Specify the file name
"chunk_number": i, # Specify the chunk number
"source": f"{os.path.basename(pdf_path)} - Chunk {i}", # Specify the chunk source
} # Define metadata for the current chunk

# Create a Document object for the current chunk
doc = Document(page_content=chunk, metadata=metadata) # Create a Document object

# Add the Document object to the list of documents
docs.append(doc) # Append the Document object to the list

# Return the list of Document objects
return docs # Return the list of Document objects
```