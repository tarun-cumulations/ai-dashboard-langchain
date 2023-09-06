from flask import Flask, request, jsonify
import requests
from werkzeug.utils import secure_filename
import traceback
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent,create_csv_agent
import pandas as pd
from dotenv import load_dotenv 
import json
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile
from tempfile import NamedTemporaryFile
import os
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import re

app = Flask(__name__)

@app.route('/generateGraphData', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("file[]")
    merge_on = request.form['merge_on']

    file_paths = []

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        filepath = os.path.join("/tmp", filename)
        file.save(filepath)
        file_paths.append(filepath)

    df = csv_tool(file_paths, merge_on)
    agent = get_conversational_agent(df)
    
    query = request.form['query']
    response = ask_agent(agent, df, query=query)
    decoded_response = decode_response(response)

    return jsonify(decoded_response)

def csv_tool(file_paths, merge_on="user_id"):
    df_list = [pd.read_csv(file) for file in file_paths]
    df_final = df_list[0]
    if len(df_list) > 1:
        for df in df_list[1:]:
            df_final = pd.merge(df_final, df, on=merge_on)
    return df_final

def get_conversational_agent(df):
    embeddings = OpenAIEmbeddings()

    text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size = 1000,
                    chunk_overlap = 200,
                    length_function=len,
    )

    texts = text_splitter.split_text(df.head().to_string())
    vectorstore = FAISS.from_texts(texts, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=vectorstore.as_retriever(),
        verbose=True
    )

    return chain


def ask_agent(agent, df, query):
    prompt = f"""
    I want to return a JSON data to a react application , it is expecting a JSON data.
    JSON data includes 1.xaxis 1.1)labelName 1.2)xAxisTickLabels 2.yaxis 2.1)labelName 2.2)yAxisTickLabels 3.typeOfGraph 4.graphData.Except this JSON response , please dont provide me anything , I want to pass this to react application , so stictly only JSON data.
    With the above instructions to be strictly followed and the following dataframe head: 
    {df.head()}
    Give me a JSON response for the following query:
    {query}
    """

    query = {
        "question": prompt,
        "chat_history": []
    }

    response = agent.run(query)
    return str(response)

def decode_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError as e:
        return response
    
@app.route('/',methods=['GET'])
def home():
    return "AI dashboard server up and running.."


@app.route('/generateKPIS', methods=['POST'])
def generate_kpis():
    uploaded_files = request.files.getlist("file[]")

    df_list = [pd.read_csv(file.stream) for file in uploaded_files]

    # Get the headers from all the dataframes
    all_headers = [df.columns.tolist() for df in df_list]

    # Flatten the list of headers and remove duplicates
    flat_headers = list(set([header for sublist in all_headers for header in sublist]))

    # Then you could ask ChatGPT to generate KPIs based on these headers
    kpis = ask_chat_gpt_to_generate_kpis(flat_headers)
    return jsonify({"kpis": kpis})

def ask_chat_gpt_to_generate_kpis(flat_headers):
    url = "https://api.openai.com/v1/chat/completions"
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {openai_api_key}"
    }
    
    data = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": f"Please respond with strictly JSON-formatted data.I am generating a thoughtful Key performance indicators which can give several insights.I will pass your response to a react application.Please strictly use JSON-formatted data.Name the JSON key as KPIs which will be Array of objects, each object containing only KPI-description.Generate KPIs based on these headers: {flat_headers}"
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        return None

if __name__ == '__main__':
    app.run(debug=True)
