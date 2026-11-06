
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# Step 1: Dataset folder location
DATASET_DIR = Path("dataset")

# Step 2: Read all text files
documents = []

for file_path in DATASET_DIR.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    if text.strip():
        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path.name}
            )
        )

if not documents:
    raise ValueError("No text files found in the dataset folder!")

print("Documents loaded:", len(documents))


# Step 3: Split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print("Total chunks created:", len(chunks))


# Step 4: Load the embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Embedding model loaded successfully!")


# Step 5: Create the ChromaDB vector database
vector_store = Chroma(
    collection_name="week11_embeddings",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# Step 6: Store embeddings in ChromaDB
# Stable IDs prevent duplicate entries when rerunning.
ids = [
    f"{chunk.metadata['source']}_{i}"
    for i, chunk in enumerate(chunks)
]

vector_store.add_documents(
    documents=chunks,
    ids=ids
)

print("Vector embeddings stored successfully!")


# Step 7: Search for relevant documents
print("\n===== SEMANTIC DOCUMENT SEARCH =====")

while True:
    query = input("\nEnter your search query (or type exit): ")

    if query.lower().strip() == "exit":
        print("Thank you!")
        break

    results = vector_store.similarity_search_with_score(
        query,
        k=3
    )

    print("\nRelevant Documents:")

    for i, (document, distance) in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("Source:", document.metadata["source"])
        print("Distance:", round(distance, 4))
        print("Content:", document.page_content)
