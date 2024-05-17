import os
import json

def read_json_files(folder_path):
    json_files = {}  
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):  
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                json_files[filename] = json.load(file)
    
    return json_files

with open('./combined_labeled_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(read_json_files('./RelevanceLabeledSentences'), f, ensure_ascii=False)