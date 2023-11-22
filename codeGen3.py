import json
from flask import Flask, request, jsonify
import pandas as pd
import openai
import re

app = Flask(__name__)

@app.route('/generate-code', methods=['POST'])
def generate_code():
    files = request.files.getlist('file')
    
    if not files:
        return jsonify({"error": "No CSV files provided"}), 400

    dataframes = [pd.read_csv(file.stream) for file in files]
    merged_df = pd.concat(dataframes, ignore_index=True)
    headers = merged_df.columns.tolist()

    query = request.form.get('user_input', '')

    messages = [
        {
            "role": "system",
            "content": "You are a Python programmer. Your task is to write Python code that manipulates a given dataframe 'df'.You should assume the 'df' will be given to you.Do not try to create a new df or modify the existing 'df'."
        },
        {
            "role": "user",
            "content": f"I need a Python script that works with an existing dataframe called 'df' with columns: {', '.join(headers)}. Please perform the following operation: {query}. Store the result of this operation in a variable named 'result'. Remember not to modify the dataframe directly and do not define a new dataframe. Please provide the complete script including any necessary import statements."
        }
    ]

    print(messages)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    
    print(response.choices[0].message['content'])
    generated_code_match = re.search(r'```python\n([\s\S]*?)\n```', response.choices[0].message['content'])
    
    if generated_code_match:
        generated_code = generated_code_match.group(1).strip()
    else:
        generated_code = response.choices[0].message['content']

    local_vars = {"df": merged_df}
    try:
        exec(generated_code, {}, local_vars)
    except Exception as e:
        return jsonify({"error": "Error executing generated code", "message": str(e)}), 500

    result = local_vars.get("result", None)
    if result is None:
        return jsonify({"error": "No result variable found in the executed code"}), 500

    try:
        if isinstance(result, (pd.DataFrame, pd.Series)):
            result = result.to_json(orient='split', index=False)  # `index=False` to ignore the index if content is empty
        else:
            result = json.dumps(result)
    except TypeError as e:
        return jsonify({"error": "Result object not JSON serializable", "message": str(e)}), 500
    
    # Now, return the serialized result
    return jsonify({"result": json.loads(result)})  # json.loads to convert the string back to a JSON object

if __name__ == "__main__":
    app.run(debug=True)
