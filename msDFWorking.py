import psycopg2
import pandas as pd
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
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from flask_cors import CORS
import random
from functools import reduce
import pyodbc

load_dotenv()

def create_db_connection(host, database, username, password):
    try:
        conn = pyodbc.connect(f'DRIVER={{/usr/lib/libmsodbcsql-17.so}};SERVER={host};DATABASE={database};UID={username};PWD={password}')
        return conn
    except Exception as e:
        print("Error while connecting to the database:", e)
        return None

def close_db_connection(conn, cursor):
    cursor.close()
    conn.close()

def fetch_dataframe_from_table(conn, table_name):
    query = f"SELECT * FROM {table_name};"
    return pd.read_sql_query(query, conn)

def get_common_column(cursor, table_names):
    common_columns = None
    for table in table_names:
        query = f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = '{table}';"
        cursor.execute(query)
        columns = {row.COLUMN_NAME for row in cursor.fetchall()}
        if common_columns is None:
            common_columns = columns
        else:
            common_columns &= columns
    return list(common_columns)[0] if common_columns else None


def merge_dataframes(conn, cursor, table_names):
    common_column = get_common_column(cursor, table_names)
    if common_column:
        df_list = [fetch_dataframe_from_table(conn, table_name) for table_name in table_names]
        df_final = df_list[0]
        for df in df_list[1:]:
            df_final = pd.merge(df_final, df, on=common_column, how='inner')
            print(df_final)
        return df_final
    else:
        print("No common column found to merge on.")
        return None


def get_table_names(cursor, schema='dbo'):  # 'dbo' is the default schema in SQL Server
    query = f"""
    SELECT table_name = t.name, schema_name = s.name
    FROM sys.tables t
    INNER JOIN sys.schemas s ON t.schema_id = s.schema_id
    WHERE s.name = '{schema}';
    """
    cursor.execute(query)
    return [row.table_name for row in cursor.fetchall()]


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
    promptForKPIs = f"""
    I want to return a JSON data to a react application , it is expecting a JSON data.
    JSON data includes 1.xaxis 1.1)labelName 1.2)xAxisTickLabels 2.yaxis 2.1)labelName 2.2)yAxisTickLabels 3.typeOfGraph 4.graphData - graphData should be strictly array of objects holding label and datapoints.Except this JSON response , please dont provide me anything , I want to pass this to react application , so stictly only JSON data.
    With the above instructions to be strictly followed and the following dataframe: 
    {df}
    Give me a JSON response for the following query:
    {query}
    """

    promptForQuestions = f"""
    I want to return a JSON data to a react application , it is expecting a JSON data.
    JSON data includes 1.TextualResponse - Answers the query or question which is a String , Apart from this dont send any extra JSON attribute..Except this JSON response , please dont provide me anything , I want to pass this to react application , so stictly only JSON data.
    With the above instructions to be strictly followed and the following dataframe : 
    {df}
    Give me a JSON response for the following query:
    {query}
    """

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
    return str(response)


def decode_response(response: str) -> dict:
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError as e:
        return response

if __name__ == "__main__":
    
    host = 'ai-dashboard-mssql.cox4boq5aldo.ap-south-1.rds.amazonaws.com'
    database = 'movies'
    username = 'admin'
    password = 'cumulations'

    conn = create_db_connection(host, database, username, password)

    if conn:
        cursor = conn.cursor()
        table_names = get_table_names(cursor)
        merged_df = merge_dataframes(conn, cursor, table_names)

        if merged_df is not None:
            agent = get_conversational_agent(merged_df)
            query = "Which movie has the highest rating"
            mode = "text"
            response = ask_agent(agent, merged_df, query=query,mode=mode)
            decoded_response = decode_response(response)
            print(decoded_response)
        else:
            close_db_connection(conn, cursor)