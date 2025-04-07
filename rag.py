import os
from uuid import uuid4
import chromadb
from chromadb.utils import embedding_functions
from chromadb import EmbeddingFunction
from typing import Optional, Union, List, Dict, Any

class RAG:
    def __init__(self, 
                 store_path: str = "", 
                 embedding_function:Optional[EmbeddingFunction] = embedding_functions.DefaultEmbeddingFunction(), 
                 persistent: bool = True,
                 chroma_executable_path: str = "chroma"):
        self.store_path = store_path
        if persistent and store_path:
            self.client = chromadb.PersistentClient(path=store_path)
        else:
            self.client = chromadb.Client()
        self.embedding_function = embedding_function
        self.chroma_executable_path = chroma_executable_path

    def check_collection(self, collection_name: str) -> bool:
        collections = self.client.list_collections()
        if collection_name in collections:
            return True
        else:
            return False
        
    def create_collection(self, collection_name: str, embedding_function:EmbeddingFunction|None = None, metadata:dict = {}) -> chromadb.Collection:
        if self.check_collection(collection_name):
            raise ValueError(f"collection {collection_name} already exists")
        if len(collection_name) < 4 or len(collection_name) > 64:
            raise ValueError("collection name should be at least 4 characters, but no more than 64 characters")
        kwargs:dict[str,Any] = {"name":collection_name}
        if embedding_function:
            kwargs["embedding_function"] = embedding_function
        if metadata:
            kwargs["metadata"] = metadata
        collection = self.client.create_collection(**kwargs)
        return collection

    def delete_collection(self, name: str):
        if not self.check_collection(name):
            raise ValueError(f"collection {name} not found")
        self.client.delete_collection(name)
        return None

    def change_collection(self, collection_name: str) -> None:
        collections = self.client.list_collections()
        if collection_name not in collections:
            raise ValueError(f"collection {collection_name} not found")
        else:
            for collection in collections:
                if collection == collection_name:
                    self.collection = self.client.get_collection(collection_name,embedding_function=self.embedding_function)
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

    def query(self, query_text: str, top_k: int = 1,similarity_value:float=0.5):
        results = self.collection.query(
            query_texts=query_text,
            n_results=top_k
        )
        for i in results:
            if i:
                pass
            else:
                raise ValueError("No result found")
        restructured = []
        for i in range(len(results['ids'][0])):
            doc_id = results['ids'][0][i]
            document = results['documents'][0][i] # type: ignore
            metadata = results['metadatas'][0][i] # type: ignore
            distance = results['distances'][0][i] # type: ignore
            similarity=(1 - abs(distance)) * 100
            if similarity<similarity_value:
                continue
            similarity=format(similarity, ".2f") + "%"
            restructured.append({
                "document": document,
                "metadata": metadata,
                "id": doc_id,
                "similarity": similarity
            })
        
        return restructured

    def update(self,id:str,text:str,metadata:dict[str,str] = {}):
        if metadata:
            self.collection.update(documents=text, metadatas=metadata, ids=id)
        else:
            self.collection.update(documents=text, ids=id)
        return None
    
    def delete(self,id:Union[str,list[str]]):
        ids = [id] if isinstance(id, str) else id
        self.collection.delete(ids=ids)
        return None

    def get_data(self):
        results=self.collection.get()
        for i in results:
            if i:
                pass
            else:
                raise ValueError("No result found")
        restructured = []
        for i in range(len(results['ids'])):
            doc_id = results['ids'][i]
            document = results['documents'][i] # type: ignore
            metadata = results['metadatas'][i] # type: ignore
            restructured.append({
                "document": document,
                "metadata": metadata,
                "id": doc_id,
            })
        return restructured

    def release_disk(self,dir:Union[str,None]=None):
        if dir:
            result=os.system(f"{self.chroma_executable_path} vacuum --path {dir} --force")
        else:
            result=os.system(f"{self.chroma_executable_path} vacuum --path {self.store_path} --force")
        if result!=0:
            raise ValueError("Failed to release disk space")
        return None