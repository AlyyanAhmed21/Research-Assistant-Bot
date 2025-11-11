# src/main.py
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.graph.graph import compile_graph

# Load environment variables
load_dotenv()

# --- API Setup ---
api = FastAPI(
    title="Research Assistant API",
    description="API for the LangGraph Research Assistant Bot.",
    version="1.0.0",
)

# --- Compile the LangGraph Agent ---
# This is done once when the API starts up
try:
    app = compile_graph()
    print("Graph compiled successfully!")
except Exception as e:
    print(f"Error compiling graph: {e}")
    app = None

# --- Pydantic Models for Request and Response ---
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    state: dict # We'll return the full final state

# --- API Endpoint ---
@api.post("/invoke", response_model=QueryResponse)
async def invoke_agent(request: QueryRequest):
    """
    Invokes the research assistant agent with a query.
    """
    if not app:
        return {"error": "Graph not compiled. Check server logs."}

    query = request.query
    if not query:
        return {"error": "Query cannot be empty."}

    inputs = {"query": query}
    final_state = app.invoke(inputs)

    return {
        "answer": final_state.get("final_answer", "No answer found."),
        "state": final_state
    }

if __name__ == "__main__":
    # This allows you to run the API server directly
    uvicorn.run(api, host="0.0.0.0", port=8000)