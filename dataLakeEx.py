import pandas as pd
from pandasai import SmartDatalake
from pandasai.llm import OpenAI

employees_data = {    'EmployeeID': [1, 2, 3, 4, 5],    'Name': ['John', 'Emma', 'Liam', 'Olivia', 'William'],    'Department': ['HR', 'Sales', 'IT', 'Marketing', 'Finance']}

salaries_data = {    'EmployeeID': [1, 2, 3, 4, 5],    'Salary': [5000, 6000, 4500, 7000, 5500]}

employees_df = pd.DataFrame(employees_data)

salaries_df = pd.DataFrame(salaries_data)

print(employees_df)

print(salaries_df)

llm = OpenAI('sk-YmucvyKpQJMnyjuXagmvTSdkDALwlQJ9sWBsFrKiJ1FuAMo3J')

dl = SmartDatalake([employees_df, salaries_df], config={"llm": llm})

print(dl.chat("Give me a bar graph for salaries and department"))

