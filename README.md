# langchain-rag

RAG solution using LangChain

## High level flow of RAG:

1. Receive an input query
2. Use the retrieval system to search for relevant information based on the query
3. Incorporate the retrieved information into the prompt sent to the LLM
4. Generate a response that leverages the retrieved context

## RAG Data Pipeline Flow

1. Documents are pushed or pulled into a data pipeline (e.g LangChain support loading documents in various formats pdf, text, docx)
2. Each document is broken down into chunks (Text splitter)

## What chunking methods should I use for the data/questions?

- **Length-based**:
  - Token-based: Splits text based on the number of tokens, which is useful when working with language models
  - Character-based: Splits text based on number of characters, which can be more consistent across different types of text (CharacterTextSplitter)
- **Text-Structured based**:
  - Splits text based on the structures such as paragraph, sentences, words (RecursiveCharacterTextSplitter)
- **Document-structure based**:
  - This chunking method is used for structured document such as **HTML**, **Markdown** and **JSON**

```
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter
# 加载 Markdown 文档
loader = UnstructuredMarkdownLoader("example.md")
documents = loader.load()
# 使用 MarkdownHeaderTextSplitter 分割
splitter = MarkdownHeaderTextSplitter(headers_to_split_on=["#", "##", "###"])
chunks = splitter.split_text(documents[0].page_content)
# 打印分割结果 for chunk in chunks:
print(chunk)
```

- **Semantic-based splitting**: This method considers the content of the text by directly analyzing the semantics of the text. The core methods relies on NLP models, using sentence embeddings (Setence-BERT) to compute semantic boundaries and detect split points through abrupt changes in text similarity.

## Embedding

Embedding models are at the core of many retrieval systems. These models transform human language into a format that machines can understand and compare quickly and accurately. They take text as input and generate a fixed-length numerical array. Embeddings enable search systems to retrieve relevant documents not only through keyword matching, but also through semantic understanding (e.g RAG)

- Reference: https://python.langchain.com/docs/concepts/embedding_models/

Key concepts:

1. Embed text as a veector: Embeddings transform text into a numerical vector representation
2. Measure similarity: Embedding vectors can be compared using simple mathematical operations

LangChain provides a unified interface for interacting with embedding models and offers standardardized methods for common operations. This general interface simplifies interactions with various embedding providers through two core methods:

- embed_documents
- embed_query

```python
from langchain_openai import OpenAIEmbeddings
# 初始化 OpenAI 嵌入模型
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
# 这里取第一个片段进行embedding，省token
document_embeddings = embeddings.embed_documents(chunks[0].page_content)
print("文档嵌入：", document_embeddings[0])
# 文档嵌入： [0.0063153719529509544, -0.00667550228536129,...]
print(len(document_embeddings))
# 935
# 单个查询
query = "这篇文章介绍DeepSeek的什么版本"
# 使用 embed_query 嵌入单个查询
query_embedding = embeddings.embed_query(query)
print("查询嵌入：", query_embedding)
# 查询嵌入：[-0.0036168822553008795, 0.0056339893490076065,...]
```

## Vector Stores

![alt text](resource\assets\image.png)

A vector store stores embedded data and performs similarity search. LangChain supports a wide varity of embedding models and vector stores. For detailed information on the supported types, refer to [full list of langchain vectorstore integrations](https://python.langchain.com/docs/integrations/vectorstores/)
