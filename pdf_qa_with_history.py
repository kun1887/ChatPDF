from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from sympy import sympify, init_printing
from sympy.parsing.latex import parse_latex
import re

file_path = input("Enter the path to the PDF file: ")
# create a vector store
loader = PyPDFLoader(file_path, extract_images=True)
pages = loader.load_and_split()
faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 10})
llm = ChatOpenAI(model="gpt-4o-mini")

# create a chain with chat history


contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(
            "chat_history"
        ),  # this placeholder will store a list of previous messages
        ("human", "{input}"),
    ]
)

# create a retriver that is history aware
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

# create a chain for question-answering tasks, with the ability to render latex code in p
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
If you have to answer a question that involves latex code, visualize it in a plain text format.\

{context}"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


# chatbot with history
def chatbot_with_history():

    chat_history = []
    print(
        "Chatbot: Hello! I'm ready to answer your questions about the pdf document. Type 'exit' to end the conversation."
    )

    while True:

        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Chatbot: Goodbye!")
            break
        ai_msg = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        chat_history.extend([HumanMessage(content=user_input), ai_msg["answer"]])
        # ai_msg["answer"] = render_math(ai_msg["answer"])
        print("Chatbot: ", end="", flush=True)
        print(ai_msg["answer"])


if __name__ == "__main__":
    chatbot_with_history()
