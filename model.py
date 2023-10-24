from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl
from flask import Flask, jsonify, request
from flask_cors import cross_origin

DB_FAISS_PATH = "vectorstore/db_faiss"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""


def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt


# Retrieval QA Chain
def retrieval_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return qa_chain


# Loading the model
def load_llm():
    # Load the locally downloaded model here
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=1024,
        temperature=0.5,
    )
    return llm


# QA Model Function
def qa_bot():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    db = FAISS.load_local(DB_FAISS_PATH, embeddings)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, db)

    return qa


Model = qa_bot()


# output function
def final_result(query):
    qa_result = qa_bot()
    response = qa_result({"query": query})
    print(response)
    return response


# chainlit code
# @cl.on_chat_start
# async def start():
#     chain = qa_bot()
#     msg = cl.Message(content="Starting the bot...")
#     await msg.send()
#     msg.content = "Hi, Welcome to GL ChatBot. What is your query?"
#     await msg.update()

#     cl.user_session.set("chain", chain)

# @cl.on_message
# async def main(message: cl.Message):
#     chain = cl.user_session.get("chain")
#     cb = cl.AsyncLangchainCallbackHandler(
#         stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
#     )
#     cb.answer_reached = True
#     res = await chain.acall(message.content, callbacks=[cb])
#     answer = res["result"]
#     sources = res["source_documents"]

#     if sources:
#         answer += f"\n \n \nSources:" + str(sources)
#     else:
#         answer += "\n \n \nNo sources found"

#     await cl.Message(content=answer).send()


# text_elements = []  # type: List[cl.Text]

# if sources:
#     for source_idx, source_doc in enumerate(sources):
#         source_name = f"source_{source_idx}"
#         # Create the text element referenced in the message
#         text_elements.append(
#             cl.Text(content=source_doc.page_content, name=source_name)
#         )
#     source_names = [text_el.name for text_el in text_elements]

#     if source_names:
#         answer += f"\nSources: {', '.join(source_names)}"
#     else:
#         answer += "\nNo sources found"
# await cl.Message(content=answer, elements=text_elements).send()


app = Flask(__name__)

# on the terminal type: curl http://127.0.0.1:5000/


@app.route("/api", methods=["GET", "POST"])
@cross_origin()
def home():
    if request.method == "GET":
        data = "hello world"
        return jsonify({"data": data})

    if request.method == "POST":
        req = request.get_json()
        print(req["question"])
        res = final_result(req["question"])
        answer = res["result"]
        return jsonify(answer)


if __name__ == "__main__":
    app.run(debug=True)
