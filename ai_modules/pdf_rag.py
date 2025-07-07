from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms import HuggingFacePipeline

# ✅ Load a local Hugging Face LLM (no API)
def get_local_llm():
    pipe = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # You can swap to bloom or mistral if needed
        max_length=512,
        do_sample=False
    )
    return HuggingFacePipeline(pipeline=pipe)

llm = get_local_llm()

def process_pdf(file_path: str, vectorstore_dir: str = "vectorstore_index"):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(vectorstore_dir)

    return vectordb

def load_existing_vectorstore(vectorstore_dir: str = "vectorstore_index"):
    embeddings = HuggingFaceEmbeddings()
    vectordb = FAISS.load_local(vectorstore_dir, embeddings, allow_dangerous_deserialization=True)
    return vectordb

def ask_question(vectordb, question: str):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=False
    )
    return qa.run(question)
