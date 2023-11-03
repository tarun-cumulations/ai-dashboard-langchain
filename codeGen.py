from flask import Flask, request, jsonify
import pandas as pd
import openai

app = Flask(__name__)

@app.route('/generate-code', methods=['POST'])
def generate_code():
    files = request.files.getlist('file')
    dataframes = {}
    for file in files:
        df = pd.read_csv(file)
        headers = df.columns.tolist()
        dataframes[file.filename] = {
            'dataframe': df,
            'headers': headers
        }

    query = request.form.get('user_input', '')
    
    final_query = f"""
    For this chat context, you are a python coder who is writing code which will be directly executed using python 'exec' function. 

    You are given a query for which you have to write a code to get the required output. 

    Assume that you have your dataframe in a variable called 'df'.

    Write a python code for the following query:
    {query}
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
    ]

    for filename, details in dataframes.items():
        headers = ', '.join(details['headers'])
        messages.append({"role": "user", "content": f"Assume you have a dataframe 'df_{filename.split('.')[0]}' with columns {headers}."})

    messages.append({"role": "user", "content": final_query})

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    generated_code = response.choices[0].message['content']

    return jsonify({"generated_code": generated_code})

if __name__ == "__main__":
    app.run(debug=True)
