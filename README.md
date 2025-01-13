# RAG (Retrieval-Augmented Generation) 项目

## 项目概述

本项目实现了一个基于检索增强生成（Retrieval-Augmented Generation, RAG）的系统，使用 `chromadb` 作为向量数据库，并结合自定义的 `ChatGLM` 嵌入模型进行文本嵌入。项目支持文本的存储、检索、以及集合的管理功能。

## 功能特性

- **文本嵌入**：使用 `ChatGLM` 模型生成文本嵌入向量。
- **集合管理**：支持创建、删除、切换集合。
- **文本存储与检索**：支持将文本及其元数据存储到集合中，并根据查询文本进行相似性检索。
- **磁盘空间释放**：支持释放集合占用的磁盘空间，重新组织数据。

## 依赖库

- `chromadb`: 向量数据库，用于存储和检索嵌入向量。
- `requests`: 用于与 `ChatGLM` API 进行通信。
- `numpy`: 用于处理嵌入向量。

## 使用方法

#### 创建集合
实例化时若collection不存在会自动创建一个collecion
```python
rag = RAG(store_path=r"D:\xxx", collection_name="my_collection",
          embedding_function=embedding_function)
```
或者使用create_collection方法创建新的集合
```python
rag.create_collection("xxx")
```

#### 删除集合
在程序结束后会自动清理磁盘文件
```python
rag.delete_collection("my_collection")
```

#### 切换集合

```python
rag.change_collection("my_collection")
```

#### 释放磁盘空间
在程序结束后会自动清理磁盘文件
```python
rag.release_disk("my_collection")
```

#### 存储文本

```python
texts = ["1", "2", "3", "4"]
metadatas = [{"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} for _ in range(len(texts))]
rag.store(texts, metadatas)
```

### 以下操作对象为实例中的collection

#### 检索文本

```python
results = rag.search("xxx", top_k=1)
```

####  获取某类数据
```python
include = ["embeddings", "documents", "metadatas"]
data = rag.get_data(include=include)
```