from medical.data_ingestion import data_ingestion
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from dotenv import load_dotenv
import os
load_dotenv()
os.environ["GROQ_API_KEY"]= os.getenv("GROQ_API_KEY")
model = ChatGroq( model="llama-3.1-70b-versatile", temperature=0.5)
chat_history = []
store = {}
def get_session_history(session_id: str, max_history_length=5) -> BaseChatMessageHistory:
  if session_id not in store:
    store[session_id] = ChatMessageHistory()
  else:
    # Limit history to the last `max_history_length` messages
    store[session_id].messages = store[session_id].messages[-max_history_length:]
  return store[session_id]
def generation(vstore):
    retriever_prompt = ("Given a chat history and the latest user question which might reference context in the chat history,"
    "formulate a standalone question which can be understood without the chat history."
    "Do NOT answer the question, just reformulate it if needed and otherwise return it as is."
    )
    retriever = vstore.as_retriever(search_kwargs={"k": 3})
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
    ("system", retriever_prompt),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ]
    )
    history_aware_retriever = create_history_aware_retriever(model, retriever, contextualize_q_prompt)
    MEDICAL_BOT_TEMPLATE = """
    You are a medical chatbot with expertise in providing accurate and helpful medical information.
    You analyze medical texts and respond to user queries based on context from the provided medical resources.
    Ensure your answers are relevant, concise, and factually accurate, based on the context provided.
    Avoid giving any advice or diagnosis that requires a licensed medical professional.
    Always remind users to consult a healthcare professional for any medical concerns.

    CONTEXT:
    {context}

    QUESTION: {input}

    YOUR ANSWER:
    """
    qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", MEDICAL_BOT_TEMPLATE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ]
)
    question_answer_chain = create_stuff_documents_chain(model, qa_prompt)
    chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


    chain_with_memmory = RunnableWithMessageHistory(
    chain,
    get_session_history,  # Pass the modified function
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
    return chain_with_memmory
if __name__ == "__main__":
   vstore = data_ingestion("done")
   conversational_rag_chain = generation(vstore)
   answer= conversational_rag_chain.invoke(
   {"input": "what is Acromegaly and gigantism?"},
    config={
        "configurable": {"session_id": "satish"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
   print(answer)
   answer1= conversational_rag_chain.invoke(
    {"input": "what is my previous question?"},
    config={
        "configurable": {"session_id": "satish"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
   print(answer1)
