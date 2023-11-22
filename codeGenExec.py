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

    # Construct messages for chat
    messages = [
        {"role": "system", "content": "You are a Python programmer who knows how to manipulate dataframes."},
        {"role": "user", "content": f"I have a dataframe called 'df' with columns: {', '.join(headers)}. {query}"}
    ]

    # Call the OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    generated_code = response.choices[0].message['content'].strip()
    print("Generated code :")
    print(generated_code)
    # Execute the generated code
    local_vars = {"df": merged_df}
    try:
        exec(generated_code, {}, local_vars)
    except Exception as e:
        return jsonify({"error": "Error executing generated code", "message": str(e)}), 500

    # Assume that the code generated a 'result' variable
    result = local_vars.get("result", "No result variable found.")

    # Return the result of the executed code as JSON
    return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=True)
