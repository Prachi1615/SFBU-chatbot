import json

conversations=[]
with open('SFBU_fine_tune_data.jsonl', 'r') as f:
    for line in f:
        
        data = json.loads(line)
        
        prompt = data.get('prompt', '')
        completion = data.get('completion', '')
        
        conversation = {
            "messages": [
                {"role": "system", "content": "You are a helpful support chatbot for San Francisco Bay University"},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": completion}
            ]
        }
        conversations.append(conversation)
        
with open('SFBU_fine_tune_data_modified.jsonl', 'w') as new_file:
    for conversation in conversations:
        json.dump(conversation, new_file)
        new_file.write('\n')