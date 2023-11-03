import json
import pandas as pd

data = {
    'address': ['123 Elm St', '456 Oak St', '789 Pine St', '101 Maple St', '202 Cedar St'],
    'salary': [50000, 60000, 55000, 62000, 51000],
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
}

df = pd.DataFrame(data)


response = {}
response['xaxis'] = {}

response['xaxis']['labelName'] = 'x-axis'
response['xaxis']['xAxisTickLabels'] = df['address'].tolist()

response['yaxis'] = {}

response['yaxis']['labelName'] = 'y-axis'

response['yaxis']['yAxisTickLabels'] = df['salary'].tolist()
 
response['typeOfGraph'] = 'bar'

response['graphData'] = []

for index, row in df.iterrows():
    data = {}
    data['label'] = row['name']  
    data['datapoints'] = [row['salary']]
    response['graphData'].append(data)

json_response = json.dumps(response)
print(json_response)