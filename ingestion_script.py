import os
import sys
# Import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# 0. Setup Paths
DOCS_PATH = './my_docs'
DB_PATH = "my_local_faiss_index"

# Check if directory exists
if not os.path.exists(DOCS_PATH):
    print(f"ERROR: The directory '{DOCS_PATH}' does not exist.")
    print("Please create a folder named 'my_docs' and put some .pdf files in it.")
    sys.exit(1)

# 1. Load Data
print(f"Loading PDFs from {DOCS_PATH}...")

# CHANGED: glob set to .pdf, and loader_cls set to PyPDFLoader
loader = DirectoryLoader(
    DOCS_PATH, 
    glob="**/*.pdf", 
    loader_cls=PyPDFLoader,
    show_progress=True
)

try:
    docs = loader.load()
except Exception as e:
    print(f"Error loading files: {e}")
    sys.exit(1)

if len(docs) == 0:
    print("ERROR: No documents found!")
    print(f"Make sure you have .pdf files inside {DOCS_PATH}")
    sys.exit(1)

print(f"Loaded {len(docs)} document pages.")

# 2. Split Data (Chunks)
print("Splitting documents...")
# PDFs often have different structures, so a slightly larger chunk size often works better
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

if len(splits) == 0:
    print("ERROR: Documents were loaded but splitting resulted in 0 chunks.")
    sys.exit(1)

print(f"Created {len(splits)} chunks.")

# 3. Embed and Store
print("Generating embeddings (this might take a moment)...")
try:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    # Test a single embedding first to catch model errors early
    test_embed = embeddings.embed_query("test")
    if not test_embed:
        print("ERROR: Ollama returned an empty embedding.")
        print("Run 'ollama pull nomic-embed-text' in your terminal.")
        sys.exit(1)
        
    vector_store = FAISS.from_documents(splits, embeddings)

    # 4. Save to disk
    vector_store.save_local(DB_PATH)
    print(f"Success! Index saved to '{DB_PATH}'")

except Exception as e:
    print(f"\nCRITICAL ERROR during embedding: {e}")
    print("1. Is Ollama running?")
    print("2. Did you run 'ollama pull nomic-embed-text'?")