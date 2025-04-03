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
from fastapi.responses import JSONResponse


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

class create_collection_data(BaseModel):
    metadata : dict = {}
@app.get("/rag/create_collection/{name}")
async def create_database(name: str,data:create_collection_data):
    if data.metadata:
        rag.create_collection(name,embedding_function=embedding_function,metadata=data.metadata)
    else:
        rag.create_collection(name,embedding_function=embedding_function)
    return JSONResponse(content={"message": f"Collection {name} created"})

@app.get("/rag/delete_collection/{name}")
async def delete_database(name: str):
    rag.delete_collection(name)
    return JSONResponse(content={"message": f"Collection {name} deleted"})

@app.get("/rag/change_collection/{name}")
async def change_database(name: str):
    rag.change_collection(name)
    return JSONResponse(content={"message": f"changed to Collection {name}"})

class store_data(BaseModel):
    text: str
    metadata: dict[str, str] = {}
@app.post("/rag/store")
async def store(data: store_data):
    print(data.text)
    print(type(data.text))
    if data.metadata:
        rag.store(text=data.text, metadata=data.metadata)
    else:
        rag.store(text=data.text)
    return JSONResponse(content={"message": "stored"})

class query_data(BaseModel):
    query_text: str
    top_k: int = 1
@app.post("/rag/query")
async def query(data: query_data):
    result = rag.query(data.query_text, top_k=data.top_k)
    return JSONResponse(content={"result": result})

class update_data(BaseModel):
    id: str
    text: str
    metadata: dict[str, str] = {}
@app.post("/rag/update")
async def update(data: update_data):
    rag.update(id=data.id, text=data.text, metadata=data.metadata)
    return JSONResponse(content={"message": "updated"})

class delete_data(BaseModel):
    id: str
@app.post("/rag/delete")
async def delete(data: delete_data):
    rag.delete(data.id)
    return JSONResponse(content={"message": "deleted"})

@app.get("/rag/release_disk/{name}")
async def release_disk(name: str):
    rag.release_disk(name)
    return JSONResponse(content={"message": f"collection {name} disk released"})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=config.server_port)