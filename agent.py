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
            temperature=settings.TEMPERATURE
        )
        self.vector_store = VectorStore()
        self.kg = KnowledgeGraphBuilder()

        self.prompt_template = PromptTemplate(
            input_variables=["context", "kg_context", "query"],
            template="""You are an expert Industrial AI Copilot. Use the following context to answer the engineer's query.
            
            Document Context:
            {context}
            
            Knowledge Graph Context (Equipment History/Topology):
            {kg_context}
            
            Query: {query}
            
            Provide a precise, technical answer. Cite your sources based on the Document Context.
            Answer:"""
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

        # 3. Generate Answer
        prompt = self.prompt_template.format(
            context=doc_context,
            kg_context=kg_context,
            query=user_text
        )
        
        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "sources": [d["metadata"]["doc_title"] for d in docs],
            "kg_data_used": bool(equipment_id and kg_data)
        }