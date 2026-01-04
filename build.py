# build.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ==========================
# Configuration
# ==========================
PDF_ROOT = "PDFs"  # Root folder containing Grade folders
PERSIST_DIR = "chroma_db"  # Where vector DB will be stored
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

# ==========================
# Helper Functions
# ==========================
def load_pdfs(pdf_root):
    documents = []
    for grade_folder in os.listdir(pdf_root):
        grade_path = os.path.join(pdf_root, grade_folder)
        if not os.path.isdir(grade_path):
            continue
        for lang_folder in os.listdir(grade_path):
            lang_path = os.path.join(grade_path, lang_folder)
            if not os.path.isdir(lang_path):
                continue
            for book_folder in os.listdir(lang_path):
                book_path = os.path.join(lang_path, book_folder)
                if not os.path.isdir(book_path):
                    continue
                for pdf_file in os.listdir(book_path):
                    if pdf_file.endswith(".pdf"):
                        pdf_path = os.path.join(book_path, pdf_file)
                        loader = PyPDFLoader(pdf_path)
                        docs = loader.load()
                        print(f"Loaded {len(docs)} pages from {pdf_path}")
                        documents.extend(docs)
    return documents

# ==========================
# Main Build
# ==========================
def main():
    print("Loading PDFs...")
    docs = load_pdfs(PDF_ROOT)

    print("Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    print("Creating embeddings...")
    embeddings = OpenAIEmbeddings()

    print("Building Chroma Vector Store...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    db.persist()
    print("Vector store built and persisted successfully!")

if __name__ == "__main__":
    main()
