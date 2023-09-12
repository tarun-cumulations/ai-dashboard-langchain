import ast

# Assuming the code snippet is in a string called code_string
code_string = '''
KPIs = [
    {'name': 'Address', 'description': 'The physical location of the employee\\'s workplace.'},
    {'name': 'Date of Joining', 'description': 'The date when the employee joined the organization.'},
    {'name': 'Name', 'description': 'The full name of the employee.'},
    {'name': 'User ID', 'description': 'A unique identifier assigned to each employee.'},
    {'name': 'Salary', 'description': 'The amount of compensation paid to the employee.'},
    {'name': 'Position', 'description': 'The job title or role of the employee within the organization.'}
]
'''

# Extract the part that defines the KPIs list
start_index = code_string.find('KPIs = [')
end_index = code_string.find(']', start_index) + 1

kpi_code = code_string[start_index:end_index]
kpi_code = kpi_code.replace('KPIs = ', '', 1)

# Evaluate the extracted code
KPIs = ast.literal_eval(kpi_code)
print(KPIs)
