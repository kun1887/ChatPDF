import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI


# ask for pdf path
file_path = input("Enter the path to the PDF file: ")

# create a vector store
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

# create a retriever
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# create a llm
llm = ChatOpenAI(model="gpt-4o-mini")


# create a conv_rag_chain
conv_rag_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
)


#
def chatbot_with_memory():
    print(
        "Chatbot: Hello! I'm ready to answer your questions about the pdf document. Type 'exit' to end the conversation."
    )

    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break

        print("Chatbot: ", end="", flush=True)
        response = conv_rag_chain.invoke({"question": user_input})
        print(response["answer"])


if __name__ == "__main__":
    chatbot_with_memory()
