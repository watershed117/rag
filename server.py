import sys
import os
# 获取当前文件所在的目录路径
try:
    cwd = os.path.dirname(os.path.abspath(__file__))
except NameError:
    cwd = os.getcwd() # Fallback

sys.path.append(cwd)
# os.chdir(cwd) # Usually not needed

import fastapi
from pydantic import BaseModel
import json
import uvicorn
from rag import RAG
from chromadb.utils.embedding_functions.openai_embedding_function import OpenAIEmbeddingFunction
from chromadb import Collection
from fastapi.responses import JSONResponse
from fastapi.responses import FileResponse

config_path = os.path.join(cwd, "config.json")
try:
    with open(config_path, "r") as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: config.json not found at {config_path}")
    sys.exit(1)
except json.JSONDecodeError:
    print(f"Error: config.json at {config_path} is not valid JSON.")
    sys.exit(1)

class Config(BaseModel):
    chroma_executable_path: str
    store_path: str
    embedding_url: str
    embedding_model: str
    embedding_api_key: str
    server_port: int

try:
    config = Config.model_validate(data)
except Exception as e:
    print(f"Error validating configuration from config.json: {e}")
    sys.exit(1)

try:
    embedding_function = OpenAIEmbeddingFunction(
                    api_key=config.embedding_api_key,
                    api_base=config.embedding_url,
                    model_name=config.embedding_model
                )
except Exception as e:
     print(f"Error initializing OpenAIEmbeddingFunction: {e}")
     sys.exit(1)

store_path_abs = os.path.join(cwd, config.store_path)
os.makedirs(store_path_abs, exist_ok=True)

try:
    rag = RAG(store_path=store_path_abs, embedding_function=embedding_function, chroma_executable_path=config.chroma_executable_path)
except Exception as e:
     print(f"Error initializing RAG with store_path='{store_path_abs}': {e}")
     sys.exit(1)

app = fastapi.FastAPI()

class create_collection_data(BaseModel):
    metadata : dict = {}

@app.post("/rag/create_collection/{name}")
async def create_database(name: str, data: create_collection_data):
    rag.create_collection(name, embedding_function=embedding_function, metadata=data.metadata)
    return JSONResponse(content={"message": f"Collection {name} created"})

@app.get("/rag/delete_collection/{name}")
async def delete_database(name: str):
    rag.delete_collection(name)
    return JSONResponse(content={"message": f"Collection {name} deleted"})

@app.get("/rag/change_collection/{name}")
async def change_database(name: str):
    rag.change_collection(name)
    return JSONResponse(content={"message": f"changed to Collection {name}"})

@app.get("/rag/list_collections")
async def list_collections():
    collections = rag.client.list_collections()
    if collections and isinstance(collections[0],Collection):
        collection_names = [c.name for c in collections]
        return JSONResponse(content={"collections": collection_names})

class store_data(BaseModel):
    text: str
    metadata: dict[str, str] = {}
@app.post("/rag/store")
async def store(data: store_data):
    if data.metadata:
        rag.store(text=data.text, metadata=data.metadata)
    else:
        rag.store(text=data.text)
    return JSONResponse(content={"message": "stored"})

class query_data(BaseModel):
    query_text: str
    top_k: int
    similarity:float=0.5
@app.post("/rag/query")
async def query(data: query_data):
    result = rag.query(data.query_text, top_k=data.top_k,similarity_value=data.similarity)
    return JSONResponse(content=result)

class update_data(BaseModel):
    id: str
    text: str
    metadata: dict
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

@app.get("/rag/get_data")
async def get_data():
    result = rag.get_data()
    return JSONResponse(content={"data": result})

class release_disk_data(BaseModel):
    path:str
@app.post("/rag/release_disk")
async def release_disk(data:release_disk_data):
    rag.release_disk(data.path)
    return JSONResponse(content={"message": f"collection {data.path} disk released"})

@app.get("/")
async def serve_frontend():
    """
    处理根路径请求，返回 static/index.html 文件。
    """
    frontend_path = os.path.join(cwd, "static", "index.html")
    if not os.path.exists(frontend_path):
         print(f"Error: Frontend file not found at {frontend_path}")
         raise fastapi.HTTPException(status_code=404, detail="Frontend file (index.html) not found in static directory.")
    return FileResponse(frontend_path, media_type='text/html')

if __name__ == "__main__":
    print(f"Starting server on http://127.0.0.1:{config.server_port}")
    print(f"Frontend should be accessible at http://127.0.0.1:{config.server_port}/")
    uvicorn.run(app, host="127.0.0.1", port=config.server_port)

