from flask import Flask, request, jsonify
import pandas as pd
import openai

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

    # We now have a single dataframe 'merged_df' that contains all the data
    headers = merged_df.columns.tolist()

    query = request.form.get('user_input', '')

    # Context for LLM assuming the merged dataframe
    context = f"You are a Python programmer. You have a DataFrame 'df' that has been merged from multiple CSV files. The DataFrame has columns: {', '.join(headers)}. "
    prompt = f"{context}Here is a query: {query}. Write the Python code to execute this query on the DataFrame."

    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", prompt=prompt)
    #response = openai.Completion.create(model="gpt-3.5-turbo", prompt=prompt)
    generated_code = response.choices[0].text.strip()

    # Execute the generated code
    local_vars = {"df": merged_df}
    exec(generated_code, {}, local_vars)
    result = local_vars.get("result", "No result variable found.")

    # Return the result of the executed code as JSON
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
