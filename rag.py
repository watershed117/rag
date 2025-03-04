from uuid import uuid4
import requests
from numpy import array
import chromadb
from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction, Embeddings
import os
from typing import Optional, Union, List, Dict, Any,Callable
import json

import subprocess
import atexit

undo_dir = []
self_path=os.path.dirname(os.path.abspath(__file__))
self_pid = os.getpid()
def cleanup():
    process = subprocess.Popen(['python', "clear.py", "--dir", str(
        undo_dir), "--pid", str(self_pid)], cwd=self_path, start_new_session=True)

atexit.register(cleanup)


class ChatGLM_EmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model: str = "embedding-3", dimensions: int = 2048):
        """
        dimensions:sugguested value in [256,512,1024,2048]
        """
        super().__init__()
        self.client = requests.Session()
        self.client.headers.update({"Authorization": f"Bearer {api_key}"})
        self.model = model
        if model == "embedding-3":
            self.dimensions = dimensions
        else:
            self.dimensions = 2048

    def __call__(self, input: str | list[str]) -> Embeddings:
        payload = {"model": self.model, "input": input,
                   "dimensions": self.dimensions}
        try:
            with self.client.post("https://open.bigmodel.cn/api/paas/v4/embeddings", json=payload) as response:
                if response.status_code == 200:
                    result = response.json()
                    embeddings = []
                    data = result["data"]
                    for embedding in data:
                        embeddings.append(array(embedding["embedding"]))
                    return embeddings
                else:
                    raise ValueError(f"failed to get embeddings from ChatGLM,status_code:{response.status_code}")
        except Exception as e:
            raise ValueError(f"failed to get embeddings from ChatGLM,error:{e}")


class RAG:
    def __init__(self, collection_name: str = "", 
                 store_path: str = "", 
                 embedding_function:Optional[EmbeddingFunction] = None, 
                 persistent: bool = True):
        self.store_path = store_path
        if persistent and store_path:
            self.client = chromadb.PersistentClient(path=store_path)
        else:
            self.client = chromadb.Client()
        self.embedding_function = embedding_function or embedding_functions.DefaultEmbeddingFunction()  # 向量维度：384
        if collection_name in self.client.list_collections():
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.create_collection(collection_name,self.embedding_function)

    def listen_folder(self, function: Callable, *args, **kwargs):
        # 获取路径下的所有文件和文件夹
        items = os.listdir(self.store_path)
        # 过滤出文件夹
        before = [item for item in items if os.path.isdir(
            os.path.join(self.store_path, item))]

        result = function(*args, **kwargs)

        # 获取路径下的所有文件和文件夹
        items = os.listdir(self.store_path)
        # 过滤出文件夹
        after = [item for item in items if os.path.isdir(
            os.path.join(self.store_path, item))]

        added = [folder for folder in after if folder not in before]
        removed = [folder for folder in before if folder not in after]
        return added, removed, result

    def create_collection(self, collection_name: str, embedding_function=embedding_functions.DefaultEmbeddingFunction()):
        if len(collection_name) < 4:
            raise ValueError("collection name should be at least 4 characters")
        collection = self.client.create_collection(
            collection_name, embedding_function=embedding_function)  # type: ignore
        ids = str(uuid4())
        added, removed, result = self.listen_folder(
            collection.add, documents="tmp", ids=ids)
        collection.delete(ids=[ids])
        with open(os.path.join(self.store_path, "config.json"), "w+") as f:
            try:
                data = json.loads(f.read())
            except:
                data = {}
        if len(added) == 1:
            data.update({collection_name: added[0]})
            with open(os.path.join(self.store_path, "config.json"), "w") as f:
                f.write(json.dumps(data))
            return collection
        else:
            self.delete_collection(collection_name)
            raise ValueError("muti floders creations detected")

    def delete_collection(self, name: str):
        with open(os.path.join(self.store_path, "config.json"), "r+") as f:
            try:
                data = json.loads(f.read())
            except:
                raise ValueError("config.json is not valid")
        dir_name = data.get(name)
        if dir_name:
            self.client.delete_collection(name)
            data.pop(name)
            with open(os.path.join(self.store_path, "config.json"), "w") as f:
                f.write(json.dumps(data))
            undo_dir.append(os.path.join(self.store_path, dir_name))
            return os.path.join(self.store_path, dir_name)
        else:
            raise ValueError(f"collection {name} not found in config.json")

    def change_collection(self, collection_name: str) -> None:
        if collection_name in self.client.list_collections():
            self.collection = self.client.get_collection(collection_name)
        else:
            self.collection = self.create_collection(collection_name,self.embedding_function)
        return None

    def store(self, 
            text: Union[str, List[str]], 
            metadata: Union[Dict[str, str], List[Dict[str, Any]]]) -> None:
        if isinstance(text, str) and isinstance(metadata, dict):
            self.collection.add(
                documents=text,
                metadatas=metadata,
                ids=str(uuid4())
            )
        elif isinstance(text, list) and isinstance(metadata, list):
            self.collection.add(
                documents=text,
                metadatas=metadata, # type: ignore
                ids=[str(uuid4()) for _ in range(len(text))]
            )
        return None

    def search(self, query_text: str, top_k: int = 1):
        results = self.collection.query(
            query_texts=query_text,
            n_results=top_k
        )
        return results["documents"], results["metadatas"], results["ids"]

    def update(self,id:str,text:str,metadata:dict[str,str] = {}):
        if metadata:
            self.collection.update(documents=text, metadatas=metadata, ids=id)
        else:
            self.collection.update(documents=text, ids=id)
        return None
    
    def get_data(self, include: list[str] = ["embeddings", "documents", "metadatas"]):
        return self.collection.get(include=include)  # type: ignore

    def release_disk(self,collection_name:str):
        data=self.get_data()
        self.delete_collection(collection_name)
        self.create_collection(collection_name,self.embedding_function)
        collection=self.client.get_collection(collection_name)
        collection.add(documents=data["documents"], metadatas=data["metadatas"], ids=data["ids"], embeddings=data["embeddings"])

if __name__ == "__main__":
    tools = [
        {
            "type": "function",
            "function": {
                "name": "query_memory",
                "description": "query memory of Ushio Noa in RAG, return text, metadata, uuid",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query_text": {
                            "type": "string",
                            "description": "the text to query",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "the number of results to return",
                            "default": 1
                        }
                    },
                    "required": ["query_text"]
                },
            }
        },
        {
            "type": "function",
            "function": {
                "name": "handle_memory",
                "description": "handle memory of Ushio Noa in RAG",
                "parameters": {
                "type": "object",
                "properties": {
                    "method": {
                    "type": "string",
                    "description": "how to handle the memory, including 'store', 'update', 'delete'",
                    "enum": ["store", "update", "delete"]
                    }
                },
                "oneOf": [
                    {
                    "properties": {
                        "method": { "const": "store" },
                        "store_text": {
                        "type": "string",
                        "description": "the text to store"
                        },
                        "store_metadata": {
                        "type": "object",
                        "description": "Metadata to associate with the stored text"
                        }
                    },
                    "required": ["store_text"]
                    },
                    {
                    "properties": {
                        "method": { "const": "update" },
                        "update_id": {
                        "type": "string",
                        "description": "the uuid of the memory to update"
                        },
                        "store_text": {
                        "type": "string",
                        "description": "the text to update"
                        },
                        "update_metadata": {
                        "type": "object",
                        "description": "Metadata to update"
                        }
                    },
                    "required": ["update_id", "store_text"]
                    },
                    {
                    "properties": {
                        "method": { "const": "delete" },
                        "delete_id": {
                        "type": "string",
                        "description": "the uuid of the memory to delete"
                        }
                    },
                    "required": ["delete_id"]
                    }
                ],
                "required": ["method"]
                }
            }
        }
    ]
    
    embedding_function = ChatGLM_EmbeddingFunction(
        api_key="", model="embedding-3", dimensions=256)
    
    embedding=embedding_function("xxx")

    rag = RAG(store_path=r"D:\xxx", collection_name="xxx",
            embedding_function=embedding_function,)

    # rag.create_collection("xxx")
    # rag.change_collection("xxx")

    import datetime
    texts = [
        "1",
        "2",
        "3",
        "4"
    ]
    metadatas = [{"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for _ in range(len(texts))]
    rag.store(texts, metadatas)
    
    print(rag.get_data())

    results = rag.search("xxx", top_k=1)
    print(results)

    rag.delete_collection(rag.collection.name)

    rag.release_disk(rag.collection.name)
