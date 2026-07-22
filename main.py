from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import IndustrialCopilotAgent
import uvicorn

from config import settings

app = FastAPI(title="Industrial Knowledge Intelligence API")

# Enable CORS for frontend and API testing tools like Postman
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

copilot = None


def get_copilot() -> IndustrialCopilotAgent:
    global copilot
    if copilot is None:
        copilot = IndustrialCopilotAgent()
    return copilot

class QueryRequest(BaseModel):
    query: str
    equipment_id: Optional[str] = None  # Optional: Pass this if asking about a specific asset

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
        result = get_copilot().query(request.query, request.equipment_id)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    """Persist an uploaded PDF under the configured data/documents folder."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file selected.")

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    documents_dir = Path(settings.DOCUMENTS_PATH)
    documents_dir.mkdir(parents=True, exist_ok=True)

    safe_name = Path(file.filename).name
    target_path = documents_dir / safe_name
    if target_path.exists():
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        target_path = documents_dir / f"{target_path.stem}_{timestamp}{target_path.suffix}"

    contents = await file.read()
    with target_path.open("wb") as handle:
        handle.write(contents)

    return {
        "success": True,
        "message": f"File uploaded successfully"
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)