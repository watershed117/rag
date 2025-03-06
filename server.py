import sys
import os
cwd=os.path.split(__file__)[0]
sys.path.append(cwd)
os.chdir(cwd)
import fastapi
from rag import RAG, SiliconFlow_EmbeddingFunction, ChatGLM_EmbeddingFunction

app = fastapi.FastAPI()

# @app.post("/rag/store")
