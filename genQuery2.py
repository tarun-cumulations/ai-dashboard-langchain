from langchain.chat_models import ChatOpenAI
import psycopg2
from dotenv import load_dotenv
import json
import os
import requests

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
    
    # You'd typically send the prompt to the chat agent here
    # For example:
    response = agent.run({"prompt": prompt, "max_tokens": 2000})
    
    # Assume response contains the SQL query
    return str(response)  # Parse this as needed

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
        
        # Step 2: User query
        user_query = "Give me the top 2 rated genre"
        
        # Step 3: Get a chat agent
        agent = get_chat_agent()
        
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
