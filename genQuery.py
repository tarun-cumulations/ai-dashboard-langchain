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

load_dotenv()

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

def get_conversational_agent():
    chain = ConversationalRetrievalChain.from_llm(
        llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
        retriever=None,  # We don't need a retriever for generating SQL queries
        verbose=True
    )

    return chain


def generate_sql_query(agent, db_context, user_query):
    prompt = f"""
    I have a database with the following tables and columns:
    Tables: {', '.join(db_context['tables'])}
    Columns: {', '.join([f"{table}: {', '.join(cols)}" for table, cols in db_context['columns'].items()])}

    I'd like to know: {user_query}
    Provide me only `SqlQuery' for this.DO NOT provide anything other than this.
    """
    
    query = {
        "question": prompt,
        "chat_history": []
    }

    response = agent.run(query)
    return str(response)  # Assume this is your SQL query; you may want to parse it further

if __name__ == "__main__":
    host = "ai-dashboard.cox4boq5aldo.ap-south-1.rds.amazonaws.com"
    database = "postgres"
    user = "postgres"
    password = "postgres"

    conn = create_db_connection(host, database, user, password)
    
    if conn:
        cursor = conn.cursor()
        
        # Step 1: Get DB context
        db_context = get_db_context(cursor)
        
        # Step 2: User query (this would come from your Flask API in a real application)
        user_query = "What is the average rating for all movies?"
        
        # Step 3: Generate SQL query using ChatGPT
        agent = get_conversational_agent()  # Passing None for now; you can pass a df if needed
        generated_sql = generate_sql_query(agent, db_context, user_query)
        
        # Step 4: Execute SQL query (after validating it to prevent SQL injection)
        cursor.execute(generated_sql)  # Make sure to validate this query before executing
        result = cursor.fetchone()
        
        print(f"Result of the query is: {result}")
        
        close_db_connection(conn, cursor)
