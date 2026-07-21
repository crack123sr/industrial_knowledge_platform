"""
Document Ingestion Pipeline
Processes PDFs and JSON documents -> chunks -> embeddings -> vector store
"""
import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime

import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

from config import settings

class DocumentProcessor:
    """Process real documents into text chunks and metadata"""
    
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
    
    def process_file(self, file_path: Path) -> Tuple[List[Dict], List[str]]:
        """
        Process a single file (PDF or JSON) into chunks
        Returns: (chunks_with_metadata, raw_chunks_list)
        """
        doc_type = file_path.suffix.lower().replace(".", "")
        full_text = ""
        doc_data = {}

        # 1. Extract raw text based on file extension
        if doc_type == "pdf":
            full_text = self._extract_from_pdf(file_path)
            doc_data = {"id": file_path.stem, "title": file_path.name}
        elif doc_type == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                doc_data = json.load(f)
            full_text = self._extract_from_json(doc_data)
            # Ensure doc_data is a dict for metadata
            if isinstance(doc_data, list):
                doc_data = {"id": file_path.stem, "title": file_path.name, "content": "List of records"}
        else:
            print(f"⚠️ Unsupported file type: {file_path.name}")
            return [], []

        if not full_text.strip():
            print(f"⚠️ No text extracted from {file_path.name}")
            return [], []

        # 2. Split into chunks
        chunks = self._chunk_text(full_text)
        
        # 3. Create chunk metadata mapping
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            chunk_meta = {
                "chunk_id": chunk_id,
                "document_id": doc_data.get("id", file_path.stem),
                "document_type": doc_type,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "text": chunk,
                "metadata": {
                    "doc_title": doc_data.get("title", file_path.name),
                    "doc_type": doc_type,
                    "ingestion_date": datetime.now().isoformat(),
                    "entities": self._extract_entities(chunk),
                }
            }
            chunk_data.append(chunk_meta)
        
        return chunk_data, chunks

    def _extract_from_pdf(self, file_path: Path) -> str:
        """Extract readable text from a real PDF document"""
        text_parts = []
        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"❌ Error reading PDF {file_path.name}: {e}")
            return ""

    def _extract_from_json(self, data: Any) -> str:
        """Extract readable text from JSON document"""
        if isinstance(data, list):
            return "\n".join([self._extract_from_json(item) for item in data])
        elif isinstance(data, dict):
            parts = []
            for key, value in data.items():
                if isinstance(value, str):
                    parts.append(f"{key}: {value}")
                elif isinstance(value, (list, dict)):
                    parts.append(f"{key}: {json.dumps(value)}")
                else:
                    parts.append(f"{key}: {str(value)}")
            return "\n".join(parts)
        return str(data)

    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks to preserve context"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.chunk_size - self.chunk_overlap):
            chunk = " ".join(words[i:i + self.chunk_size])
            if chunk:
                chunks.append(chunk)
        
        return chunks if chunks else [text]

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Simple entity extraction using keyword matching"""
        entities = {
            "equipments": [],
            "actions": [],
        }
        equipment_keywords = ["pump", "valve", "compressor", "motor", "sensor", "transmitter", "vessel"]
        action_keywords = ["repair", "replacement", "maintenance", "inspection", "calibration", "testing"]
        
        text_lower = text.lower()
        
        for keyword in equipment_keywords:
            if keyword in text_lower:
                entities["equipments"].append(keyword)
        for keyword in action_keywords:
            if keyword in text_lower:
                entities["actions"].append(keyword)
                
        return entities


class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        self.dimension = settings.FAISS_DIMENSION
        self.index = None
        self.metadata_store = []  # Store chunk metadata alongside vectors
        self.load_or_create_index()
    
    def load_or_create_index(self):
        """Load existing index or create new one"""
        index_dir = Path(settings.FAISS_INDEX_PATH)
        index_dir.mkdir(parents=True, exist_ok=True)
        
        index_path = index_dir / "faiss.index"
        metadata_path = index_dir / "metadata.json"
        
        if index_path.exists() and metadata_path.exists():
            print(f"📚 Loading existing FAISS index from {index_path}")
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'r') as f:
                self.metadata_store = json.load(f)
        else:
            print(f"🆕 Creating new FAISS index")
            self.index = faiss.IndexFlatL2(self.dimension)
    
    def add_documents(self, chunks: List[Dict[str, Any]]):
        """Convert chunks to vectors and add to FAISS"""
        if not chunks:
            return
        
        texts = [chunk["text"] for chunk in chunks]
        
        print(f"  🧮 Embedding {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts, show_progress_bar=False)
        embeddings = embeddings.astype(np.float32)
        
        # Add to FAISS and metadata store
        self.index.add(embeddings)
        self.metadata_store.extend(chunks)
        
        print(f"  ✅ Added {len(chunks)} chunks. Total in store: {len(self.metadata_store)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search for relevant document chunks"""
        if self.index.ntotal == 0:
            return []

        query_embedding = self.embedding_model.encode([query], show_progress_bar=False)[0]
        query_embedding = np.array([query_embedding], dtype=np.float32)
        
        distances, indices = self.index.search(query_embedding, min(top_k, len(self.metadata_store)))
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:
                continue
            
            chunk = self.metadata_store[int(idx)]
            results.append({
                "chunk_id": chunk["chunk_id"],
                "document_id": chunk["document_id"],
                "text": chunk["text"],
                "similarity_score": float(1 / (1 + distances[0][i])),
                "metadata": chunk["metadata"],
            })
        
        return results
    
    def save(self):
        """Persist index and metadata to disk"""
        index_path = Path(settings.FAISS_INDEX_PATH)
        faiss.write_index(self.index, str(index_path / "faiss.index"))
        
        with open(index_path / "metadata.json", 'w') as f:
            json.dump(self.metadata_store, f, indent=2, default=str)
        print(f"💾 Saved FAISS index and metadata")


class IngestionPipeline:
    """End-to-end document ingestion manager"""
    
    def __init__(self):
        self.processor = DocumentProcessor()
        self.vector_store = VectorStore()
    
    def ingest_all_documents(self):
        """Scan the documents folder and ingest everything inside it"""
        print("\n🔄 Starting document ingestion pipeline...")
        
        docs_path = Path(settings.DOCUMENTS_PATH)
        
        # Look for both PDFs and JSONs
        files_to_process = list(docs_path.glob("*.pdf")) + list(docs_path.glob("*.json"))
        
        if not files_to_process:
            print(f"❌ No documents found in {docs_path}. Drop some PDFs or JSONs in there first.")
            return
        
        total_chunks = 0
        
        for file_path in files_to_process:
            print(f"\n📖 Processing {file_path.name}...")
            chunks, _ = self.processor.process_file(file_path)
            
            if chunks:
                self.vector_store.add_documents(chunks)
                total_chunks += len(chunks)
        
        # Save results to disk
        self.vector_store.save()
        print(f"\n✅ Ingestion complete! Total chunks embedded: {total_chunks}")
        print(f"📊 Total documents in vector store: {len(self.vector_store.metadata_store)}")


# ==========================================
# Run this file directly to test the pipeline
# ==========================================
if __name__ == "__main__":
    pipeline = IngestionPipeline()
    
    # 1. Run the ingestion process
    pipeline.ingest_all_documents()
    
    # 2. Test a sample search
    print("\n-------------------------------------------")
    print("Testing Vector Search:")
    test_query = "What are the maintenance steps?"
    print(f"Query: '{test_query}'")
    
    results = pipeline.vector_store.search(test_query, top_k=2)
    if not results:
        print("No results found. (Did you add documents to the data/documents folder?)")
    else:
        for r in results:
            doc_title = r['metadata']['doc_title']
            preview = r['text'][:150].replace('\n', ' ')
            print(f"  -> Found in [{doc_title}]: {preview}...")