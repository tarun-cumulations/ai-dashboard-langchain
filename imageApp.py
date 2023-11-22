from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
from dotenv import load_dotenv 
from pandasai import SmartDatalake
from pandasai.llm import OpenAI

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/get-analytics-csv', methods=['GET'])
def get_analytic_csv():
    employees_data = {    'EmployeeID': [1, 2, 3, 4, 5],    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],    'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']}

    salaries_data = {    'EmployeeID': [1, 2, 3, 4, 5],    'Salary': [5000, 6000, 4500, 7000, 5500]}

    employees_df = pd.DataFrame(employees_data)

    salaries_df = pd.DataFrame(salaries_data)

    llm = OpenAI('sk-YmucvyKpQJMnyjuXagmvTSdkDALwlQJ9sWBsFrKiJ1FuAMo3J')

    dl = SmartDatalake([employees_df, salaries_df], config={"llm": llm})

    print(dl.chat("Give me a bar graph for salaries and department"))

    return "jsssjj"


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)