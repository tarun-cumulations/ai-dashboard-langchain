import json
from flask import Flask, request, jsonify
import pandas as pd
import openai
import re

app = Flask(__name__)

@app.route('/generate-code', methods=['POST'])
def generate_code():
    files = request.files.getlist('file')
    
    # If no files are provided, return an error
    if not files:
        return jsonify({"error": "No CSV files provided"}), 400

    # Load the CSV files into dataframes
    dataframes = [pd.read_csv(file.stream) for file in files]

    # Merge the dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    headers = merged_df.columns.tolist()

    query = request.form.get('user_input', '')

    # messages = [
    #     {"role": "system", "content": "You are a Python programmer."},
    #     {"role": "user", "content": f"Write a Python script that operates on a dataframe 'df' with columns: {', '.join(headers)}. The script should store the result of the operation in a variable named 'result'. The operation to perform is as follows: {query}.Strictly do not create or define your own df,assume you have a df passed to you in variable called 'df'.While fetching the result ignore the index if content is empty."}
    # ]

    messages = [
    {
        "role": "system",
        "content": "You are a Python programmer. Your task is to write Python code that manipulates a given dataframe 'df'."
    },
    {
        "role": "user",
        "content": f"I need a Python script that works with an existing dataframe called 'df'. This dataframe has columns: {', '.join(headers)}. Please perform the following operation: {query}. Store the result of this operation in a variable named 'result'. Remember not to modify the dataframe directly and do not define a new dataframe. Please provide the complete script including any necessary import statements."
    }
]


    # messages = [
    #     {"role": "system", "content": "You are a Python programmer."},
    #     {"role": "user", "content": f"Provide a Python script that operates on a csv content with columns: {', '.join(headers)}. The script should store the result of the operation in a variable named 'result'. The operation to perform is as follows: {query}."}
    # ]

    print(messages)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    print(response.choices[0].message['content'])
    # Check if the response contains markdown code block
    generated_code_match = re.search(r'```python\n([\s\S]*?)\n```', response.choices[0].message['content'])
    
    if generated_code_match:
        # If markdown is found, extract the code
        generated_code = generated_code_match.group(1).strip()
    else:
        # Otherwise, use the content directly
        generated_code = response.choices[0].message['content']
    # Execute the extracted code
    local_vars = {"df": merged_df}
    try:
        print("Executing code")
        exec(generated_code, {}, local_vars)
    except Exception as e:
        return jsonify({"error": "Error executing generated code", "message": str(e)}), 500

    # Retrieve the 'result' variable from local_vars
    result = local_vars.get("result", None)
    if result is None:
        return jsonify({"error": "No result variable found in the executed code"}), 500

    # Return the result as part of the response
    #return jsonify({"result": result})

    if not isinstance(result, str):
        try:
            # If result is a pandas object, convert to JSON-compatible format first
            if isinstance(result, (pd.DataFrame, pd.Series)):
                result = result.to_json(orient='split')
            else:
                result = json.dumps(result)
        except TypeError as e:
            return jsonify({"error": "Result object not JSON serializable", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
