from uuid import uuid4
import requests
from numpy import array
import chromadb
from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction, Embeddings
import os
from typing import Optional, Union, List, Dict, Any,Callable
import json
from requests import RequestException,ConnectionError,HTTPError

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
    def __init__(self, api_key: str, model: str = "embedding-3", dimensions: int = 2048, timeout: int = 10):
        """
        dimensions:sugguested value in [256,512,1024,2048]
        """
        super().__init__()
        self.client = requests.Session()
        self.client.headers.update({"Authorization": f"Bearer {api_key}"})
        self.model = model
        self.timeout = timeout
        if model == "embedding-3":
            self.dimensions = dimensions
        else:
            self.dimensions = 2048

    def __call__(self, input: str | list[str]) -> Embeddings:
        payload = {"model": self.model, "input": input,
                   "dimensions": self.dimensions}
        try:
            with self.client.post("https://open.bigmodel.cn/api/paas/v4/embeddings", json=payload, timeout=self.timeout) as response:
                if response.status_code == 200:
                    result = response.json()
                    embeddings = []
                    data = result["data"]
                    for embedding in data:
                        embeddings.append(array(embedding["embedding"]))
                    return embeddings
                else:
                    raise HTTPError(response=response)
        except requests.Timeout:
            raise ConnectionError("Request timed out")
        except ConnectionError as e:
            raise ConnectionError(f"Network connection error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")

class SiliconFlow_EmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, model: str = "BAAI/bge-m3", timeout: int = 10):
        """
        dimensions:sugguested value in [256,512,1024,2048]
        """
        super().__init__()
        self.client = requests.Session()
        self.client.headers.update({"Authorization": f"Bearer {api_key}"})
        self.model = model
        self.timeout = timeout

    def __call__(self, input: str | list[str]) -> Embeddings:
        payload = {"model": self.model, "input": input, "encoding_format": "float"}
        try:
            with self.client.post("https://api.siliconflow.cn/v1/embeddings", json = payload, timeout = self.timeout) as response:
                if response.status_code == 200:
                    result = response.json()
                    embeddings = []
                    data = result["data"]
                    for embedding in data:
                        embeddings.append(array(embedding["embedding"]))
                    return embeddings
                else:
                    raise HTTPError(response=response)
        except requests.Timeout:
            raise ConnectionError("Request timed out")
        except ConnectionError as e:
            raise ConnectionError(f"Network connection error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")
    
class Online_EmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: str, url:str="https://api.siliconflow.cn/v1/embeddings", model: str = "BAAI/bge-m3" , timeout: int = 10):
        super().__init__()
        self.client = requests.Session()
        self.client.headers.update({"Authorization": f"Bearer {api_key}"})
        self.url = url
        self.model = model
        self.timeout = timeout

    def __call__(self, input: str | list[str], **kwargs) -> Embeddings:
        payload = {"model": self.model, "input": input}
        if kwargs:
            payload.update(kwargs)
        try:
            with self.client.post(self.url, json = payload, timeout = self.timeout) as response:
                if response.status_code == 200:
                    result = response.json()
                    embeddings = []
                    data = result["data"]
                    for embedding in data:
                        embeddings.append(array(embedding["embedding"]))
                    return embeddings
                else:
                    raise HTTPError(response=response)
        except requests.Timeout:
            raise ConnectionError("Request timed out")
        except ConnectionError as e:
            raise ConnectionError(f"Network connection error: {e}")
        except Exception as e:
            raise RuntimeError(f"Unexpected error: {e}")
        
class RAG:
    def __init__(self, 
                 store_path: str = "", 
                 embedding_function:Optional[EmbeddingFunction] = None, 
                 persistent: bool = True):
        self.store_path = store_path
        if persistent and store_path:
            self.client = chromadb.PersistentClient(path=store_path)
        else:
            self.client = chromadb.Client()
        self.embedding_function = embedding_function or embedding_functions.DefaultEmbeddingFunction()  # 向量维度：384

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

    def create_collection(self, collection_name: str, embedding_function:EmbeddingFunction|None = None, metadata:dict = {}) -> chromadb.Collection:
        if len(collection_name) < 4:
            raise ValueError("collection name should be at least 4 characters")
        kwargs:dict[str,Any] = {"name":collection_name}
        if embedding_function:
            kwargs["embedding_function"] = embedding_function
        if metadata:
            kwargs["metadata"] = metadata
        collection = self.client.create_collection(**kwargs)
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
            self.collection = self.client.get_collection(collection_name,embedding_function=self.embedding_function) # type: ignore
        else:
            raise ValueError(f"collection {collection_name} not found")
        return None

    def store(self, 
            text: Union[str, List[str]], 
            metadata: Union[Dict[str, str], List[Dict[str, Any]],None]=None) -> None:
        kwargs:dict[str,Any]={"documents":text}
        if metadata and isinstance(metadata, dict):
            kwargs["metadatas"]=[metadata]
        if isinstance(text, str):
            kwargs["ids"]=[str(uuid4())]
        if isinstance(text, list):
            kwargs["ids"]=[str(uuid4()) for _ in range(len(text))]
        self.collection.add(**kwargs)
        return None

    def query(self, query_text: str, top_k: int = 1):
        results = self.collection.query(
            query_texts=query_text,
            n_results=top_k
        )
        return results["documents"], results["metadatas"], results["ids"], results["distances"]

    def update(self,id:str,text:str,metadata:dict[str,str] = {}):
        if metadata:
            self.collection.update(documents=text, metadatas=metadata, ids=id)
        else:
            self.collection.update(documents=text, ids=id)
        return None
    
    def delete(self,id:str|list[str]):
        ids = [id] if isinstance(id, str) else id
        self.collection.delete(ids=ids)
        return None

    def get_data(self, include: list[str] = ["embeddings", "documents", "metadatas"]):
        return self.collection.get(include=include)  # type: ignore

    def release_disk(self,collection_name:str):
        data=self.get_data()
        self.delete_collection(collection_name)
        self.create_collection(collection_name,self.embedding_function)
        collection=self.client.get_collection(collection_name)
        collection.add(documents=data["documents"], metadatas=data["metadatas"], ids=data["ids"], embeddings=data["embeddings"])


