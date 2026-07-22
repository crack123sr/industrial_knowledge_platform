from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import IndustrialCopilotAgent
import uvicorn

app = FastAPI(title="Industrial Knowledge Intelligence API")

# Enable CORS for frontend and API testing tools like Postman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

copilot = IndustrialCopilotAgent()

from typing import Optional  # Make sure to import Optional

class QueryRequest(BaseModel):
    query: str
    equipment_id: Optional[str] = None  # Allows string, null, or missing entirely

@app.get("/")
def health_check():
    return {"status": "Operational", "system": "Industrial Knowledge Brain"}

@app.post("/api/copilot/query")
def ask_copilot(request: QueryRequest):
    """
    Receives JSON input containing a 'query' and optional 'equipment_id',
    passes it to the IndustrialCopilotAgent, and returns the response.
    """
    try:
        # Pass data from the Pydantic model directly to the agent
        result = copilot.query(request.query, request.equipment_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)