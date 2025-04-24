import os
import sys
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA
from langchain_community.cache import InMemoryCache
from langchain.globals import set_llm_cache


# Use the OPENAI_API_KEY to initialize the OpenAI API Client
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_TEST_KEY")
os.environ['OPENAI_ORG_ID'] = os.getenv("OPENAI_ORG_ID")


def load_documents(file_path):
    """
    Load document under the given path based on file type

    Args:
        file_path (_type_): _description_

    Returns:
        document
    """
    # Load documents from a directory
    if file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    elif file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.docx'):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please provide a .txt, .pdf, or .docx file.")
    document = loader.load()
    return document

def load_directory(directory_path):
    """
    Load documents from a directory

    Args:
        directory_path (_type_): _description_

    Returns:
        documents
    """
    documents = []

    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        if os.path.isfile(file_path) and not filename.endswith(('.faiss','.pkl')):
            try:
                document = load_documents(file_path)
                documents.extend(document)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
    return documents

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into chunks

    Args:
        documents (_type_): _description_
        chunk_size (int, optional): _description_. Defaults to 1000.
        chunk_overlap (int, optional): _description_. Defaults to 200.

    Returns:
        chunks
    """
    # Split the documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

def create_vector_store(chunks, save_path):
    """
    Create a vector store from the chunks of text

    Args:
        chunks (_type_): _description_
        save_path (_type_): _description_

    Returns:
        vector_store
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)
    # Create a vector store from the chunks of text
    vector_store = FAISS.from_documents(documents =chunks, embedding=embeddings)
    
    if save_path:
        vector_store.save_local(save_path)
        print(f"Vector store saved to {save_path}")
    return vector_store

def load_vector_store(vector_store_path):
    """
    Load a vector store from the given path

    Args:
        vector_store_path (_type_): _description_

    Returns:
        vector_store
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002", chunk_size=1000)
    # Load the vector store from the given path
    vector_store = FAISS.load_local(
        folder_path=vector_store_path, 
        embeddings=embeddings, 
        index_name="index", 
        allow_dangerous_deserialization=True
        )
    print(f"Loaded {len(vector_store.docstore._dict)} documents from {vector_store_path}")
    return vector_store

def setup_qa_chain(vector_store):
    """
    Setup the QA chain with the given vector store

    Args:
        vector_store (_type_): _description_

    Returns:
        qa_chain
    """

    # Initialize an LLM tool - OpenAI GPT-3.5
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # Create a multi-query retriever
    multi_retriever = MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

    set_llm_cache(InMemoryCache())
    # Create a RetrievalQA chain
    # The chain will use the retriever to find relevant documents and the LLM to generate answers
    # The chain will return the answer and the source documents
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=multi_retriever, chain_type="stuff", return_source_documents=True)    
    
    return qa_chain

def ask_question(qa_chain, query):
    """_summary_

    Args:
        qa_chain (_type_): _description_
        query (_type_): 
        
    """ 
    response = qa_chain({"query": query})
    return response

def main():
    # Check if the script is run directly
    if len(sys.argv) < 2:
        print("Usage: python Langchain_qa.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]
    vector_store_path = os.path.join(directory_path, "vector_store")
    
    # Load documents from the directory
    documents = load_directory(directory_path)
    
    # Split documents into chunks
    chunks = split_documents(documents)
    
    # Create a vector store from the chunks
    vector_store = create_vector_store(chunks, vector_store_path)
    
    # Load the vector store from the given path
    vector_store = load_vector_store(vector_store_path)
    
    # Setup the QA chain with the vector store
    qa_chain = setup_qa_chain(vector_store)
    
    print("\n=== 文档问答系统已启动 ===")
    print("输入问题进行查询，输入 'exit' 退出")
    
    while True:
        query = input("\n请输入问题: ")
        if query.lower() in ['exit', 'quit', '退出']:
            break
            
        try:
            answer = ask_question(qa_chain, query)
            print("\n回答:")
            print(answer['result'])
        except Exception as e:
            print(f"查询出错: {str(e)}")
    
    print("感谢使用，再见！")

if __name__ == "__main__":
    main()