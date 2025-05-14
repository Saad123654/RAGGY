#imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings

print ("Imports réussis")

###########################################################################
# Import du llm
llm = OllamaLLM(model="mistral")


print ("LLM importé")

###########################################################################
# setup du système de RAG
# Function to set up the RAG system
def setup_rag_system():
    # Load the document
    loader = TextLoader('data/my_document.txt')
    documents = loader.load()
    # Import de l'embedding
    embeddings = OllamaEmbeddings(model="mistral")
    # Split the document into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    document_chunks = splitter.split_documents(documents)    
    # Create FAISS vector store from document chunks and embeddings
    vector_store = FAISS.from_documents(document_chunks, embeddings)

    # Return the retriever for document retrieval with specified search_type
    retriever = vector_store.as_retriever(
        search_type="similarity",  # or "mmr" or "similarity_score_threshold"
        search_kwargs={"k": 5}  # Adjust the number of results if needed
    )
    return retriever

print ("RAG system configuré")
#####################################################################################

# Function to get the response from the RAG system
def get_rag_response(query: str):
    retriever = setup_rag_system()
    # Retrieve the relevant documents using 'get_relevant_documents' method
    retrieved_docs = retriever.invoke(query)
    # Prepare the input for the LLM: Combine the query and the retrieved documents into a single string
    context = "\n".join([doc.page_content for doc in retrieved_docs])
    # LLM expects a list of strings (prompts), so we create one by combining the query with the retrieved context
    prompt = [f"Use the following information to answer the question:\n\n{context}\n\nQuestion: {query}"]
    # Generate the final response using the language model (LLM)
    generated_response = llm.generate(prompt)  # Pass as a list of strings    
    return generated_response

print ("Fonction de réponse RAG configurée")
#############################################################################################################

print('test....')


# Example usage
if __name__ == "__main__":
    # Visualize the chunks
    retriever = setup_rag_system()
    print("RAG system is set up and ready to use.")
    # Print the first few chunks for verificationdocs = retriever.vectorstore.docstore._dict.values()

    docs = retriever.vectorstore.docstore._dict.values()

    for i, doc in enumerate(docs):
        print(f"Chunk {i+1}:\n{doc.page_content}")
        print("-" * 80)

    # Get a query from the user
    query = input("Please enter your query: ")  # Capture the query from the user

    # Get a response from the RAG system
    response = get_rag_response(query)
    print("Response:", response)
