import os
from pypdf import PdfReader

def load_docs(folder_path):
    documents = []
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.endswith(".txt"):
            with open(file_path, "r") as f:
                text = f.read()
            documents.append({
                "filename": filename,
                "text": text
            })
        
        elif filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            documents.append({
                "filename": filename,
                "text": text
            })
    
    return documents