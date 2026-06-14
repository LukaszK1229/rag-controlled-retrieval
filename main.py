from fastapi import FastAPI
from pydantic import BaseModel
import query_updated

app = FastAPI(title="Controlled Retrieval API")


class QueryRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query")
def query_rag(req: QueryRequest):

    result = query_updated.run_query(req.question)

    return result