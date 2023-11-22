import base64
import requests
import os 
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('OPENAI_API_KEY')

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

image_path = "/home/tarun/cumulations/experiments/sales-dash/langchain_test-main/images/highest_salary_employee.png"

base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4-vision-preview",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "From this image , I want to return a JSON data to a react application, it is expecting a JSON data.JSON data includes 1.xaxis 1.1)labelName 1.2)xAxisTickLabels 2.yaxis 2.1)labelName 2.2)yAxisTickLabels 3.typeOfGraph 4.graphData.Apart from this, don't send any extra JSON attribute. Except this JSON response, please dont provide me anything.I want to pass this to a react application, so strictly only JSON data."
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 2000
}

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

print(response.json()['choices'][0]['message']['content'].strip())