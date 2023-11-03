from flask import Flask, request, jsonify
import requests
from werkzeug.utils import secure_filename
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent,create_csv_agent
import pandas as pd
from dotenv import load_dotenv 
import json
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
from flask_cors import CORS
import random
from functools import reduce

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/get-analytics', methods=['POST'])
def upload_files():
    uploaded_files = request.files.getlist("file[]")
    #merge_on = request.form['merge_on']
    mode = request.form['mode']

    file_paths = []
    column_names_list = []

    for file in uploaded_files:
        filename = secure_filename(file.filename)
        filepath = os.path.join("/tmp", filename)
        file.save(filepath)
        file_paths.append(filepath)
        df_for_merge_on_field = pd.read_csv(filepath, nrows=1)
        column_names_list.append(set(df_for_merge_on_field.columns))

    
    common_columns = reduce(lambda x, y: x.intersection(y), column_names_list)
    print("Column column is :",common_columns)

    if not common_columns:
        return jsonify({"error": "No common columns to merge on.Either provide a single CSV file or provide CSV files which have common columns to merge"}), 400

    # Choose the first common column to merge on
    merge_on = list(common_columns)[0]

    df = csv_tool(file_paths, merge_on)
    agent = get_conversational_agent(df)

    
    query = request.form['query']
    response = ask_agent(agent, df, query=query,mode=mode)
    decoded_response = decode_response(response)

    return jsonify(decoded_response)

def csv_tool(file_paths, merge_on="user_id"):
    df_list = [pd.read_csv(file) for file in file_paths]
    df_final = df_list[0]
    if len(df_list) > 1:
        for df in df_list[1:]:
            df_final = pd.merge(df_final, df, on=merge_on)
    print(df_final)
    return df_final

def get_conversational_agent(df):
    embeddings = OpenAIEmbeddings()

    text_splitter = CharacterTextSplitter(
                    separator="\n",
                    chunk_size = 10000,
                    chunk_overlap = 200,
                    length_function=len,
    )

    texts = text_splitter.split_text(df.to_string())
    vectorstore = FAISS.from_texts(texts, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=vectorstore.as_retriever(),
        verbose=True
    )

    return chain


def ask_agent(agent, df, query , mode):
    promptForKPIs = """
    For this chat context, you are a python coder who is writing code which will be directly executed using python 'exec' function. 

    - Write code assuming you have a dataframe with a variable 'df'.
    - Generate a JSON response based on the query.
    
    The JSON response should have the following structure:
    1. 'xaxis'
        1.1 'labelName'
        1.2 'xAxisTickLabels'
    2. 'yaxis'
        2.1 'labelName'
        2.2 'yAxisTickLabels'
    3. 'typeOfGraph'
    4. 'graphData': An array of objects, each holding 'label' and 'datapoints'.

    I will pass you the dataframe head , use this as reference on how the df may look.

    This is the dataframe head : {df.head.to_string()}

    Note: Apart from the JSON output, the code should not return anything else.

    Here's your query to execute:
    {query}

    Return the python code that you wrote in 'code' as JSON response , which i will execute with exec function.
    """


    print(promptForKPIs)

    promptForQuestions = """
    For this chat context, you are a Python coder. Your code will be executed directly using Python's 'exec' function. Here's what you need to do:

    - Assume you have a dataframe with a variable 'df'.
    - Write code to generate a JSON response based on the query.

    The JSON response should have the following structure:
    1. 'TextualResponse': A string that answers the query or question.

    Note: Apart from this JSON response, the code should not return anything else.

    Here's your query to execute:
    {query}

    Return the Python code that you wrote in 'code' as a JSON response, which I will execute using the exec function.Apart from code , DO NOT return me anything.
    """

    print(promptForQuestions)

    finalPrompt = ""

    if(mode=="text"):
        finalPrompt = promptForQuestions
    else:
        finalPrompt = promptForKPIs

    query = {
        "question": finalPrompt,
        "chat_history": []
    }

    response = agent.run(query)
    print("response :"+response)
    return str(response)

def decode_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError as e:
        return response
    
@app.route('/',methods=['GET'])
def home():
    return "AI dashboard server up and running.."


@app.route('/generate-kpi-and-questions', methods=['POST'])
def generate_kpis():
    try:
        uploaded_files = request.files.getlist("file[]")
        df_list = [pd.read_csv(file.stream) for file in uploaded_files]
        all_headers = [df.columns.tolist() for df in df_list]
        flat_headers = list(set([header for sublist in all_headers for header in sublist]))

        kpis_str = ask_chat_gpt_to_generate_kpis(flat_headers)
        questions = ask_chat_gpt_to_generate_questions(flat_headers)
        kpis_dict = json.loads(kpis_str)
        ques_dict = json.loads(questions)
        print(ques_dict)
        descriptions = [kpi['description'] for kpi in kpis_dict.get('KPIs', [])]

        enhanced_descriptions = []
        for desc in descriptions:
           enhanced_descriptions.append(f"{random.choice(['Give me a bar graph for', 'Give me a line graph for'])} {desc}")

        question_list = [q['question'] for q in ques_dict.get('Questions', [])]

        return jsonify({"KPIs": enhanced_descriptions, "Questions": question_list})

    except KeyError as e:
        return jsonify({"error": f"KeyError: {e}"})

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {e}"})
    

def ask_chat_gpt_to_generate_questions(flat_headers):
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
                "content": f"Please respond with strictly JSON-formatted data.I am generating a thoughtful questions on this dataset.I will pass your response to a react application.Please strictly use JSON-formatted data.Name the JSON key as Questions which will be Array of objects containing only question.Generate thoughtful questions by which I can get some meaningful insights based on these headers: {flat_headers}"
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        decoded_response = decode_response(response.json()['choices'][0]['message']['content'])
        return json.dumps(decoded_response, default=str)
    else:
        return None

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
                "content": f"Please respond with strictly JSON-formatted data.I am generating a thoughtful Key performance indicators which can give several insights.I will pass your response to a react application.Please strictly use JSON-formatted data.Name the JSON key as KPIs which will be Array of objects, each object containing only description.Generate KPIs based on these headers: {flat_headers}"
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        decoded_response = decode_response(response.json()['choices'][0]['message']['content'])
        return json.dumps(decoded_response, default=str)
    else:
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)