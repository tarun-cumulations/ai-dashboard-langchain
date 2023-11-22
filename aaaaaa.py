from flask import Flask, request, jsonify
import pandas as pd
import openai
import re

app = Flask(__name__)

@app.route('/generate-code', methods=['POST'])
def generate_code():
    files = request.files.getlist('file')
    dataframes = [pd.read_csv(file) for file in files]

    # Check if there are any dataframes to merge
    if not dataframes:
        return jsonify({"error": "No files provided"}), 400

    # Merge the dataframes
    merged_df = pd.concat(dataframes, ignore_index=True)
    headers = merged_df.columns.tolist()

    query = request.form.get('user_input', '')

    messages = [
        {"role": "system", "content": "You are a Python programmer."},
        {"role": "user", "content": f"Store the final answer in a variable called 'result'.I have a dataframe called 'df' with columns: {', '.join(headers)}. {query}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    

    print(response.choices[0].message['content'])
    # Extract code from the response using regex
    generated_code_match = re.search(r'```python\n([\s\S]*?)\n```', response.choices[0].message['content'])

    if not generated_code_match:
        return jsonify({"error": "Generated code format is incorrect"}), 500

    generated_code = generated_code_match.group(1).strip()

    

    # Execute the extracted code
    local_vars = {"df": merged_df,"result":""}
    try:
        print("Exceuting generated code")
        exec(generated_code, {}, local_vars)
        print("Result :",result)
    except Exception as e:
        return jsonify({"error": "Error executing generated code", "message": str(e)}), 500

    # Assume that the code generated a 'result' variable
    result = local_vars.get("result", None)
    if result is None:
        return jsonify({"error": "No result variable found in the executed code"}), 500


if __name__ == "__main__":
    app.run(debug=True)