import re
import json
import pandas as pd
from ollama import chat, ChatResponse

# Sample input
model_name = "ds0"

data_file_path = 'D:/HackOs3/hackos-3/data/actual/dataset.csv'
df = pd.read_csv(data_file_path, sep="|")
#df = df.iloc[:20]
error = 0
correct = 0
results =[]

# Function to extract JSON from response
def extract_json(response_text):
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            print("Error: Failed to parse JSON.")
    return {}

for i, row in df.iterrows():
    prompt = row["input"]

    response: ChatResponse = chat(model=model_name, messages=[{"role": "user", "content": prompt}])

    response_json = response.json()
    response_text = response["message"]["content"]
    extracted_data = extract_json(response_text)

    # Ensure required fields exist
    parsed_data = {
        "input": prompt,
        "error_type": extracted_data.get("error_type", ""),
        "severity": extracted_data.get("severity", ""),
        "description": extracted_data.get("description", ""),
        "solution": extracted_data.get("solution", ""),
    }

    results.append(parsed_data)

    if parsed_data["severity"] == df.iloc[i]["severity"]:
        if parsed_data["error_type"] == df.iloc[i]["error_type"]:
            correct += 1
        else:
            error +=1 
    else:
        error +=1 

    print('.')


print("Accuracy = " + str((correct)/(correct + error)))
df_dump = pd.DataFrame(results)
df_dump.to_csv('dump.csv', index = False)
    
