"""
LLM Knowledge Graph Extractor
Standalone script to read an industrial PDF manual, use Gemini to extract 
entities and relationships, and generate a structured JSON file.
"""

import os
import sys
import json
from pathlib import Path
import PyPDF2
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# Import settings from your existing config
from config import settings

class GraphExtractor:
    def __init__(self):
        # Initialize Gemini with a low temperature for strict factual output
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1 
        )
        
        self.prompt_template = PromptTemplate(
            input_variables=["filename", "text"],
            template="""
            You are an expert Industrial Knowledge Extraction AI.
            Analyze the following text from an industrial manual ({filename}) and extract the equipment, components, maintenance tasks, safety rules, and their relationships.
            
            Output STRICTLY as a valid JSON array of objects matching this exact schema:
            [
              {{
                "id": "Name of equipment, component, or task",
                "type": "Equipment" | "Component" | "MaintenanceTask" | "Problem" | "Safety",
                "properties": {{"description": "Brief description", "spec": "Any technical spec if available"}},
                "relations": [
                    {{"target": "Related Entity ID", "relation_type": "PART_OF" | "REQUIRES_MAINTENANCE" | "CAUSED_BY" | "RESOLVED_BY" | "HAS_COMPONENT"}}
                ]
              }}
            ]
            
            Extract a comprehensive core summary graph (10 to 20 highly relevant entities). 
            Do NOT include markdown formatting like ```json. Return ONLY the raw, valid JSON.
            
            TEXT TO ANALYZE:
            {text}
            """
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Reads text from the PDF file."""
        print(f"📄 Reading PDF: {pdf_path.name}...")
        text_parts = []
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                # Limit to the first 30 pages to avoid exceeding LLM context windows 
                # for standard API tiers, while capturing the core technical data.
                num_pages = min(len(reader.pages), 30)
                for i in range(num_pages):
                    text = reader.pages[i].extract_text()
                    if text:
                        text_parts.append(text)
            return "\n".join(text_parts)
        except Exception as e:
            print(f"❌ Error reading PDF: {e}")
            return ""

    def generate_graph_json(self, pdf_path_str: str):
        """Main pipeline to extract text, call LLM, and save JSON."""
        pdf_path = Path(pdf_path_str)
        
        if not pdf_path.exists() or pdf_path.suffix.lower() != '.pdf':
            print(f"❌ Invalid file path or not a PDF: {pdf_path_str}")
            return

        # 1. Get Text
        raw_text = self.extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            print("❌ No text could be extracted from the PDF.")
            return

        # Limit text size to prevent token overload (approx 60,000 chars)
        processed_text = raw_text[:60000]

        # 2. Call Gemini
        print("🧠 Sending text to Gemini for ontology extraction. Please wait...")
        try:
            prompt = self.prompt_template.format(filename=pdf_path.name, text=processed_text)
            response = self.llm.invoke(prompt)
            
            # Clean up the output to ensure valid JSON
            clean_json_str = response.content.replace('```json', '').replace('```', '').strip()
            graph_data = json.loads(clean_json_str)
            
            # 3. Save Output
            output_filename = f"{pdf_path.stem}_graph.json"
            output_path = Path(settings.DOCUMENTS_PATH) / output_filename
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(graph_data, f, indent=2)
                
            print(f"✅ Success! Extracted {len(graph_data)} entities.")
            print(f"💾 Graph JSON saved to: {output_path}")
            
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse Gemini's response into JSON. Error: {e}")
            print("Raw Output was:\n", clean_json_str)
        except Exception as e:
            print(f"❌ An error occurred during LLM processing: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python llm_graph_extractor.py <path_to_pdf>")
        print("Example: python llm_graph_extractor.py data/documents/t999_user_manual.pdf")
    else:
        extractor = GraphExtractor()
        extractor.generate_graph_json(sys.argv[1])