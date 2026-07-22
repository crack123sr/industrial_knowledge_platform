import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from config import settings
from ingestion import VectorStore
from kg_builder import KnowledgeGraphBuilder

class IndustrialCopilotAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.1
        )
        self.vector_store = VectorStore()
        self.kg = KnowledgeGraphBuilder()

        # DYNAMIC GENERAL PROMPT: Adapts to any query type
        self.prompt_template = PromptTemplate(
            input_variables=["context", "kg_context", "query"],
            template="""You are an expert Industrial AI Copilot. 
            Analyze the user's query and use the Document Context (PDF chunks) and Knowledge Graph Context provided below to construct a comprehensive answer.
            
            Document Context:
            {context}
            
            Knowledge Graph Context:
            {kg_context}
            
            User Query: {query}
            
            Return your response STRICTLY as a valid JSON object. Do not include markdown formatting like ```json. Use this schema:
            {{
                "query_type": "The category of the query (e.g., Safety, Maintenance, Troubleshooting, Specification, General)",
                "summary": "A concise overview answering the query",
                "extracted_text_chunks": [
                    "Direct text reference or step 1 from documents",
                    "Direct text reference or step 2 from documents"
                ],
                "knowledge_graph_relations": [
                    "Relevant entity or connection found in the graph, if any"
                ],
                "recommended_actions": [
                    "Actionable step or recommendation for the technician"
                ]
            }}
            """
        )

    def query(self, user_text: str, equipment_id: str = None) -> dict:
        # 1. Vector Search (FAISS)
        docs = self.vector_store.search(user_text, top_k=settings.TOP_K_DOCUMENTS)
        doc_context = "\n".join([f"Source ({d['metadata']['doc_title']}): {d['text']}" for d in docs])

        # 2. Graph Search (NetworkX)
        kg_context = "No specific equipment ID provided."
        if equipment_id:
            kg_data = self.kg.get_equipment_context(equipment_id)
            kg_context = str(kg_data) if kg_data else f"No data found for {equipment_id}."
        else:
            # Try a fuzzy search using query terms if no explicit ID was passed
            matched_nodes = self.kg.search_entities(user_text)
            if matched_nodes:
                kg_data = self.kg.get_equipment_context(matched_nodes[0])
                kg_context = str(kg_data)

        # 3. Generate Answer
        prompt = self.prompt_template.format(
            context=doc_context,
            kg_context=kg_context,
            query=user_text
        )
        
        response = self.llm.invoke(prompt)

        # 4. Parse the LLM output into a clean Python dictionary
        try:
            clean_json = response.content.replace('```json', '').replace('```', '').strip()
            structured_response = json.loads(clean_json)
        except json.JSONDecodeError:
            structured_response = {
                "query_type": "General",
                "summary": response.content,
                "extracted_text_chunks": [],
                "knowledge_graph_relations": [],
                "recommended_actions": []
            }

        return {
            "response": structured_response,
            "sources_used": [d["metadata"]["doc_title"] for d in docs],
            "kg_data_used": bool(kg_context != "No specific equipment ID provided.")
        }