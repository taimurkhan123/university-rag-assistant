import os
import shutil
import gdown
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

GDRIVE_FOLDER_ID = "1Ijz0ogF_rpaiKhM8QM7TkiihA39YPjFP"
LOCAL_DATA_DIR = "./university_data"
FAISS_DB_PATH = "faiss_index"

def download_drive_folder(folder_id, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading files from Google Drive (ID: {folder_id})...")
    gdown.download_folder(id=folder_id, output=output_dir, quiet=False, use_cookies=False)
    print("Download completed.")

def build_faiss_index():
    print("Loading documents...")
    loader = DirectoryLoader(LOCAL_DATA_DIR, glob="*.txt", loader_cls=TextLoader)
    documents = loader.load()
    
    if not documents:
        raise ValueError("No text documents found in downloaded folder.")

    print(f"Loaded {len(documents)} document(s). Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print("Generating vector embeddings using HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(FAISS_DB_PATH)
    print(f"FAISS index created successfully at '{FAISS_DB_PATH}'.")

if __name__ == "__main__":
    download_drive_folder(GDRIVE_FOLDER_ID, LOCAL_DATA_DIR)
    build_faiss_index()
