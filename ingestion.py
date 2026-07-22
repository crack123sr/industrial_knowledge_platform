"""
Document Ingestion Pipeline
Processes PDFs and JSON documents -> chunks -> embeddings -> vector store
"""
import os
import json
import hashlib
import re
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
        """
        Split text into chunks with AGGRESSIVE segmentation.
        Prioritizes creating many smaller chunks over perfect context preservation.
        """
        text = text.strip()
        if not text:
            return []
        
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strategy 1: Force split on page boundaries
        pages = text.split('Page ')
        if len(pages) > 2:  # More than 1 actual page marker
            page_chunks = []
            for page in pages:
                if page.strip():
                    page_chunks.extend(self._aggressively_chunk(page))
            return page_chunks if page_chunks else self._aggressively_chunk(text)
        
        # Strategy 2: Split aggressively
        return self._aggressively_chunk(text)

    def _aggressively_chunk(self, text: str) -> List[str]:
        """Aggressively split text into small chunks"""
        # Strategy A: Split by section headers (numbered or CAPS)
        sections = self._extract_sections(text)
        if len(sections) > 2:
            return self._chunk_sections_small(sections)
        
        # Strategy B: Split by paragraphs
        paragraphs = self._split_into_paragraphs(text)
        if len(paragraphs) > 3:
            return self._chunk_paragraphs_small(paragraphs)
        
        # Strategy C: Split by lines (most aggressive)
        lines = self._split_into_lines(text)
        if len(lines) > 10:
            return self._chunk_lines_small(lines)
        
        # Strategy D: Fallback to character chunking
        return self._chunk_by_characters(text)

    def _extract_sections(self, text: str) -> List[str]:
        """Extract sections by detecting headers and structural breaks"""
        # Match: "1 SECTION NAME", "2.1 SUBSECTION", "##Section", etc.
        section_pattern = r'(?:^|\n)(?:\d+(?:\.\d+)?\s+[A-Z][^\n]*|^[A-Z][A-Z\s]{5,}[^\n]*\n)'
        
        sections = []
        current_section = []
        
        for line in text.split('\n'):
            # Check if this line is a header
            if re.match(r'^\d+(?:\.\d+)?\s+[A-Z]|^[A-Z][A-Z\s]{5,}', line.strip()):
                if current_section:
                    sections.append('\n'.join(current_section))
                    current_section = [line]
                else:
                    current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append('\n'.join(current_section))
        
        return [s.strip() for s in sections if len(s.strip()) > 30]

    def _chunk_sections_small(self, sections: List[str]) -> List[str]:
        """Chunk sections with SMALL limit - force splits"""
        chunks = []
        
        for section in sections:
            section_lines = section.split('\n')
            
            # If section is long, break it further
            if len(section_lines) > 15:
                chunks.extend(self._chunk_lines_small(section_lines))
            else:
                chunks.append(section)
        
        return [c.strip() for c in chunks if len(c.strip()) > 30]

    def _chunk_paragraphs_small(self, paragraphs: List[str]) -> List[str]:
        """Chunk paragraphs with aggressive word limit"""
        chunks = []
        current_chunk = []
        current_words = 0
        
        # Use SMALL word limit for aggressive chunking
        word_limit = max(150, self.chunk_size // 2)
        
        for para in paragraphs:
            para_words = len(para.split())
            
            # If para alone is small enough, add it
            if para_words < word_limit:
                if current_words + para_words < word_limit:
                    current_chunk.append(para)
                    current_words += para_words
                else:
                    # Start new chunk
                    if current_chunk:
                        chunks.append('\n'.join(current_chunk))
                    current_chunk = [para]
                    current_words = para_words
            else:
                # Para itself is too big, split it
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_words = 0
                
                # Split large para by sentences
                sentences = self._split_into_sentences(para)
                temp_chunk = []
                temp_words = 0
                
                for sent in sentences:
                    sent_words = len(sent.split())
                    if temp_words + sent_words < word_limit:
                        temp_chunk.append(sent)
                        temp_words += sent_words
                    else:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [sent]
                        temp_words = sent_words
                
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [c.strip() for c in chunks if len(c.strip()) > 30]

    def _chunk_lines_small(self, lines: List[str]) -> List[str]:
        """Chunk lines VERY aggressively - small line groups"""
        chunks = []
        current_chunk = []
        current_words = 0
        
        # Very small limit for lines
        word_limit = max(100, self.chunk_size // 3)
        
        for line in lines:
            if not line.strip():
                continue
            
            line_words = len(line.split())
            
            if current_words + line_words < word_limit:
                current_chunk.append(line)
                current_words += line_words
            else:
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_words = line_words
        
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [c.strip() for c in chunks if len(c.strip()) > 30]

    def _chunk_by_characters(self, text: str) -> List[str]:
        """
        Character-based chunking as absolute fallback.
        Guaranteed to chunk any text regardless of structure.
        """
        if not text:
            return []
        
        # For very long text, use character-based chunking
        char_chunk_size = self.chunk_size * 5  # ~5 chars per word
        overlap_size = self.chunk_overlap * 5
        
        chunks = []
        step = max(1, char_chunk_size - overlap_size)
        
        for i in range(0, len(text), step):
            chunk = text[i:i + char_chunk_size]
            if chunk.strip():
                chunks.append(chunk)
        
        return chunks if chunks else [text]

    def _split_into_paragraphs(self, text: str) -> List[str]:
        """Split text by double newlines or section breaks"""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_into_lines(self, text: str) -> List[str]:
        """Split text by single newlines"""
        lines = text.split('\n')
        return [line.strip() for line in lines if line.strip()]

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using multiple patterns"""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\n|(?<=\))\s+[A-Z]'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _chunk_by_sections(self, sections: List[str]) -> List[str]:
        """Create chunks from major sections, combining smaller ones if needed"""
        if not sections:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for section in sections:
            section_length = len(section.split())
            
            # If section + current would exceed limit, start new chunk
            if current_length + section_length > self.chunk_size and current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append(chunk_text)
                current_chunk = [section]
                current_length = section_length
            else:
                current_chunk.append(section)
                current_length += section_length
        
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        
        return chunks if chunks else sections[:1] if sections else []

    def _chunk_by_paragraphs(self, paragraphs: List[str]) -> List[str]:
        """Create chunks by combining paragraphs to reach chunk_size"""
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para.split())
            
            # If adding this paragraph exceeds chunk_size, start new chunk
            if current_length + para_length > self.chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep last paragraph for overlap
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add remaining
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks if chunks else paragraphs

    def _chunk_by_lines(self, lines: List[str]) -> List[str]:
        """Create chunks by combining lines - more aggressive for technical docs"""
        if not lines:
            return []
        
        # For technical docs: use smaller effective chunk size
        effective_chunk_size = max(100, self.chunk_size // 2)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line.split())
            
            # If adding this line exceeds effective chunk size, start new chunk
            if current_length + line_length > effective_chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep last 2-3 lines for overlap
                overlap_lines = min(3, len(current_chunk) // 3)
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else [line]
                current_length = len("\n".join(current_chunk).split())
            
            current_chunk.append(line)
            current_length += line_length
        
        # Add remaining
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks if chunks else lines

    def _chunk_by_sentences(self, sentences: List[str]) -> List[str]:
        """Create chunks by combining sentences to reach chunk_size"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_buffer = []
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, start new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Prepare overlap: keep last N sentences
                words_in_chunk = len(chunk_text.split())
                overlap_words = min(self.chunk_overlap, words_in_chunk)
                
                # Find how many sentences fit in overlap
                overlap_buffer = []
                temp_length = 0
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if temp_length + sent_words <= overlap_words:
                        overlap_buffer.insert(0, sent)
                        temp_length += sent_words
                    else:
                        break
                
                current_chunk = overlap_buffer.copy()
                current_length = len(" ".join(current_chunk).split())
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [" ".join(sentences)]
        """Create chunks by combining paragraphs to reach chunk_size"""
        if not paragraphs:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for para in paragraphs:
            para_length = len(para.split())
            
            # If adding this paragraph exceeds chunk_size, start new chunk
            if current_length + para_length > self.chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep last paragraph for overlap
                current_chunk = [para]
                current_length = para_length
            else:
                current_chunk.append(para)
                current_length += para_length
        
        # Add remaining
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks if chunks else paragraphs

    def _chunk_by_lines(self, lines: List[str]) -> List[str]:
        """Create chunks by combining lines - more aggressive for technical docs"""
        if not lines:
            return []
        
        # For technical docs: use smaller effective chunk size
        effective_chunk_size = max(100, self.chunk_size // 2)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line.split())
            
            # If adding this line exceeds effective chunk size, start new chunk
            if current_length + line_length > effective_chunk_size and current_chunk:
                chunk_text = "\n".join(current_chunk)
                chunks.append(chunk_text)
                
                # Keep last 2-3 lines for overlap
                overlap_lines = min(3, len(current_chunk) // 3)
                current_chunk = current_chunk[-overlap_lines:] if overlap_lines > 0 else [line]
                current_length = len("\n".join(current_chunk).split())
            
            current_chunk.append(line)
            current_length += line_length
        
        # Add remaining
        if current_chunk:
            chunks.append("\n".join(current_chunk))
        
        return chunks if chunks else lines

    def _chunk_by_sentences(self, sentences: List[str]) -> List[str]:
        """Create chunks by combining sentences to reach chunk_size"""
        if not sentences:
            return []
        
        chunks = []
        current_chunk = []
        current_length = 0
        overlap_buffer = []
        
        for sentence in sentences:
            sentence_length = len(sentence.split())
            
            # If adding this sentence exceeds chunk_size, start new chunk
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Finalize current chunk
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                
                # Prepare overlap: keep last N sentences
                words_in_chunk = len(chunk_text.split())
                overlap_words = min(self.chunk_overlap, words_in_chunk)
                
                # Find how many sentences fit in overlap
                overlap_buffer = []
                temp_length = 0
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if temp_length + sent_words <= overlap_words:
                        overlap_buffer.insert(0, sent)
                        temp_length += sent_words
                    else:
                        break
                
                current_chunk = overlap_buffer.copy()
                current_length = len(" ".join(current_chunk).split())
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks if chunks else [" ".join(sentences)]

    def _chunk_by_words(self, text: str) -> List[str]:
        """Fallback word-based chunking for short text or edge cases"""
        words = text.split()
        if not words:
            return []
        
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            if chunk_words:
                chunks.append(" ".join(chunk_words))
            i += step
        
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