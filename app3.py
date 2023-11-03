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
import psycopg2




app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/get-analytics', methods=['POST'])
def get_analyticc():
    mode = request.form['mode']

    print("Called get analytics API ")
    # connect to database and perform db query
    host = "ai-dashboard.cox4boq5aldo.ap-south-1.rds.amazonaws.com"
    database = "postgres"
    user = "postgres"
    password = "postgres"

    conn = create_db_connection(host, database, user, password)
    
    if conn:
        print("Connections is established with the DB")
        cursor = conn.cursor()
        
        # Step 1: Get DB context
        db_context = get_db_context(cursor)
        
        # Step 2: User query
        #user_query = "Give me the top 2 rated genre"
        user_query = request.form['query']
        
        print("agent called")
        # Step 3: Get a chat agent
        agent = get_chat_agent()
        
        print("Agent call id =s finished")
        # Step 4: Generate SQL query
        generated_sql = ask_chat_gpt_for_sql_query(db_context, user_query)
        
        print("Answer query")
        print(generated_sql)

        #clean extra text and get the query
        start_keyword = "sql\n" 
        end_keyword = ";\n"

        start_index = generated_sql.find(start_keyword) + len(start_keyword)
        end_index = generated_sql.find(end_keyword)

        extracted_sql = generated_sql[start_index:end_index].strip()
        print(extracted_sql)
        # Step 5: Execute SQL query (after validating it)
        cursor.execute(extracted_sql)

        result1 = cursor.fetchall()
        
       
        print(f"Result of the query using fetchall: {result1}")
        
        close_db_connection(conn, cursor)
    ##
    #answer = "[('Crime', Decimal('9.5000000000000000')), ('Sci-Fi', Decimal('9.5000000000000000'))]"
    answer = str(result1)

    agent = get_conversational_agent(answer)
    response = ask_agent(agent,mode=mode,answer=answer)
    decoded_response = decode_response(response)

    return jsonify(decoded_response)


def get_conversational_agent(answer):
    embeddings = OpenAIEmbeddings()

    texts = [answer]
    vectorstore = FAISS.from_texts(texts, embeddings)

    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=vectorstore.as_retriever(),
        verbose=True
    )

    return chain


def ask_agent(agent, mode, answer=""):
    if mode == "text":
        finalPrompt = f"""
        I want to return a JSON data to a react application, it is expecting a JSON data.
        JSON data includes 1.TextualResponse - Answers the query or question which is a String. 
        Apart from this, don't send any extra JSON attribute. Except this JSON response, please don't provide me anything;
        I want to pass this to a react application, so strictly only JSON data.
        The answer to the query is: 
        {answer}
        """
    else:
        finalPrompt = f"""
        I want to return a JSON data to a react application, it is expecting a JSON data.
        JSON data includes 1.xaxis 1.1)labelName 1.2)xAxisTickLabels 2.yaxis 2.1)labelName 2.2)yAxisTickLabels 3.typeOfGraph 4.graphData.
        Apart from this, don't send any extra JSON attribute. Except this JSON response, please don't provide me anything;
        I want to pass this to a react application, so strictly only JSON data.
        The answer to the query is: 
        {answer}
        """

    query = {
        "question": finalPrompt,
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
    
def create_db_connection(host, database, user, password):
    try:
        conn = psycopg2.connect(
            host=host,
            database=database,
            user=user,
            password=password
        )
        return conn
    except Exception as e:
        print("Error while connecting to the database:", e)
        return None

def close_db_connection(conn, cursor):
    cursor.close()
    conn.close()

def get_db_context(cursor):
    table_names = get_table_names(cursor)
    db_context = {"tables": table_names, "columns": {}}
    
    for table in table_names:
        query = f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table}';"
        cursor.execute(query)
        db_context["columns"][table] = [column[0] for column in cursor.fetchall()]
        
    return db_context

def get_table_names(cursor, schema='public'):
    query = f"""
    SELECT table_name
    FROM information_schema.tables
    WHERE table_schema = '{schema}';
    """
    cursor.execute(query)
    return [table[0] for table in cursor.fetchall()]

def get_chat_agent():
    return ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo')

def ask_chat_gpt_for_sql_query(db_context, user_query):
    url = "https://api.openai.com/v1/chat/completions"
    openai_api_key = os.environ.get('OPENAI_API_KEY')
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {openai_api_key}"
    }
    
    prompt = f"""
    I have a database with the following tables and columns:
    Tables: {', '.join(db_context['tables'])}
    Columns: {', '.join([f"{table}: {', '.join(cols)}" for table, cols in db_context['columns'].items()])}

    I'd like to know: {user_query}
    Can you give me an SQL query to get this information?
    """

    print("prompt :",prompt)
    
    data = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        generated_sql = response.json()['choices'][0]['message']['content'].strip()
        return generated_sql  # Parse this as needed
    else:
        return None

def generate_sql_query(agent, db_context, user_query):
    prompt = f"""
    I have a database with the following tables and columns:
    Tables: {', '.join(db_context['tables'])}
    Columns: {', '.join([f"{table}: {', '.join(cols)}" for table, cols in db_context['columns'].items()])}

    I'd like to know: {user_query}
    Can you give me an SQL query to get this information?
    """
    
    response = agent.run({"prompt": prompt, "max_tokens": 2000})
    
    
    return str(response)  # Parse this as needed


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