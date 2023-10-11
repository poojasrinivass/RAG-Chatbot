import os
import sys
# import pinecone
from langchain.llms import Replicate
from langchain.vectorstores import Pinecone
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS

# Replicate API token to access llama2
os.environ['REPLICATE_API_TOKEN'] = "r8_cAuNFO7H08S8DwoZfUczo5AJaq5AmlT2AEGKm"


def get_textchunks(pdf):
    loader = PyPDFLoader(pdf)
    documents = loader.load()
    # Split the documents into smaller chunks for processing
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def get_vectorstore(text_chunks):
    #creating a vector index for the document
    embeddings = HuggingFaceEmbeddings()
    vectordb  = FAISS.from_documents(text_chunks, embeddings)
    return vectordb

def get_conv_chain(vectordb):
    # Initialize Replicate Llama2 Model
    llm = Replicate(
        model="a16z-infra/llama13b-v2-chat:df7690f1994d94e96ad9d568eac121aecf50684a0b0963b25a41cc40061269e5",
        input={"temperature": 0.75, "max_length": 3000}
    )

    chat_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectordb.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True)
    return chat_chain

if __name__ == "__main__":
    text_chunks = get_textchunks('Knowledge_document.pdf')
    vectordb = get_vectorstore(text_chunks)
    qa_chain = get_conv_chain(vectordb)
    chat_history = []
    while True:
        query = input('Prompt: ')
        if query.lower() in ["exit", "quit", "q"]:
            print('Exiting')
            sys.exit()
        result = qa_chain({'question': query, 'chat_history': chat_history})
        print('Answer: ' + result['answer'] + '\n')
        chat_history.append((query, result['answer']))
