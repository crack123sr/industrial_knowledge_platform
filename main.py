from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import IndustrialCopilotAgent
import uvicorn

app = FastAPI(title="Industrial Knowledge Intelligence API")
copilot = IndustrialCopilotAgent()

class QueryRequest(BaseModel):
    query: str
    equipment_id: str = None  # Optional: Pass this if the user is asking about a specific asset

@app.get("/")
def health_check():
    return {"status": "Operational", "system": "Industrial Knowledge Brain"}

@app.post("/api/copilot/query")
def ask_copilot(request: QueryRequest):
    try:
        result = copilot.query(request.query, request.equipment_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)