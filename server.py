import sys
import os
cwd=os.path.split(__file__)[0]
sys.path.append(cwd)
os.chdir(cwd)
import fastapi
from pydantic import BaseModel
import json
import uvicorn
from rag import RAG, Online_EmbeddingFunction

with open("config.json", "r") as f:
    data = json.load(f)

class Config(BaseModel):
    store_path: str
    embedding_url: str
    embedding_model: str
    embedding_api_key: str
    server_port: int

config=Config.model_validate(data)
embedding_function = Online_EmbeddingFunction(api_key=config.embedding_api_key, url=config.embedding_url, model=config.embedding_model)
rag = RAG(store_path=config.store_path, embedding_function=embedding_function)
app = fastapi.FastAPI()

@app.get("/rag/create_collection/{name}")
async def create_database(name: str):
    rag.create_collection(name)
    return {"message": f"Collection {name} created"}

@app.get("/rag/delete_collection/{name}")
async def delete_database(name: str):
    rag.delete_collection(name)
    return {"message": f"Collection {name} deleted"}

@app.get("/rag/change_collection/{name}")
async def change_database(name: str):
    rag.change_collection(name)
    return {"message": f"Collection {name} changed"}


@app.post("/rag/store")
async def store(text: str, metadata: dict[str, str] = {}):
    rag.store(text=text, metadata=metadata)
    return {"message": "stored"}

@app.post("/rag/query")
async def query(query_text: str, top_k: int = 1):
    result = rag.query(query_text, top_k=top_k)
    return {"result": result}

@app.post("/rag/update")
async def update(id: str, text: str, metadata: dict[str, str] = {}):
    rag.update(id=id, text=text, metadata=metadata)
    return {"message": "updated"}

@app.post("/rag/delete")
async def delete(id: str):
    rag.delete(id)
    return {"message": "deleted"}

@app.get("/rag/release_disk/{name}")
async def release_disk(name: str):
    rag.release_disk(name)
    return {"message": f"collection {name} disk released"}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=config.server_port)