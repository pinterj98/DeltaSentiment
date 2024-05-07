import os
from bs4 import BeautifulSoup
import json
import nltk
import json
from functools import reduce
from nltk.tokenize import sent_tokenize
from time import time

nltk.download('punkt')

def read_html_files(folder_path):
    html_files = {}  
    for filename in os.listdir(folder_path):
        if filename.endswith(".html"):  
            file_path = os.path.join(folder_path, filename)  
            with open(file_path, 'r', encoding='utf-8') as file:
                html_files[filename] = BeautifulSoup(file, "html.parser")
    return html_files


def get_paragraphs(html_soup):
    paragraphs_with_min_words = []
    min_word_count = 8

    for paragraph in html_soup.find_all('p'):
        words = paragraph.text.split()
        if len(words) >= min_word_count:
            paragraphs_with_min_words.append(paragraph.text)

    return paragraphs_with_min_words

def save_paragraphs(paragraphs):
    for name, paragraph in paragraphs.items():
        with open('./Paragraphs/'+name, 'w', encoding='utf-8') as f:
            json.dump(paragraph, f, ensure_ascii=False)


def get_sentences(paragraph_list):
    sentences = reduce(lambda x,y: x+sent_tokenize(y), [[]]+paragraph_list)
    return sentences

def save_sentences(sentences):
    for name, sentence_list in sentences.items():
        with open('./Sentences/'+name, 'w', encoding='utf-8') as f:
            json.dump(sentence_list, f, ensure_ascii=False)

t1=time()

html_data = read_html_files('./HTMLData')
paragraphs = {x.replace('.html', '_paragraphs.json') : get_paragraphs(html_data[x]) for x in html_data}
sentences = {x.replace('_paragraphs.json', '_sentences.json') : get_sentences(paragraphs[x]) for x in paragraphs}

save_paragraphs(paragraphs)
save_sentences(sentences)

sentences = {x.replace('.json', '') : sentences[x] for x in sentences}

with open('./combined_sentences.json', 'w', encoding='utf-8') as f:
    json.dump(sentences, f, ensure_ascii=False)

t2=time()

print(f'Process finished in {t2-t1} seconds.')
