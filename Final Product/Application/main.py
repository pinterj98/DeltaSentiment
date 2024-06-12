import streamlit as st
from bs4 import BeautifulSoup
import os
import json
import nltk
import json
from functools import reduce
from nltk.tokenize import sent_tokenize
from time import time
import io
import base64
import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
import plotly.express as px


st.set_page_config(
    layout="wide"
)


@st.cache_data
def load_torch():
    model_name = "yiyanghkust/finbert-tone"
    st.session_state['model'] =  torch.load('FullTrain2best')
    st.session_state['tokenizer'] =  AutoTokenizer.from_pretrained(model_name)
    st.session_state['collator'] = DataCollatorWithPadding(st.session_state['tokenizer'])
    return None


@st.cache_data
def load_torch2():
    model_name = "yiyanghkust/finbert-tone"
    st.session_state['model2'] =  AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    return None

load_torch()
load_torch2()

class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.float).unsqueeze(0)
        }


def predict(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.sigmoid(logits).cpu().numpy()
            predictions.extend(preds)
    return [x[0] for x in predictions]

def predict2(model, dataloader):
    predictions = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to('cuda')
            attention_mask = batch['attention_mask'].to('cuda')
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.softmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)
    return [x[1]-x[2] for x in predictions]


if "button1" not in st.session_state:
    st.session_state["button1"] = False

if "button2" not in st.session_state:
    st.session_state["button2"] = False

if "button3" not in st.session_state:
    st.session_state["button3"] = False


css='''
<style>
    section.main > div {max-width:75rem}
    .big-font {
    font-size:18px;
    margin-bottom:-50px;
    }
    .stApp {background-color:#f7f7f7}
    header[data-testid="stHeader"] {background-color:#f7f7f7}
</style>
'''
st.markdown(css, unsafe_allow_html=True)

# Download link
def create_download_link(my_list, text_to_show, filename="my_list.json"):
    # Convert list to JSON
    json_str = json.dumps(my_list, indent=4, ensure_ascii=False)
    # Encode JSON string to base64
    b64 = base64.b64encode(json_str.encode()).decode()
    # Create download link
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">{text_to_show}</a>'
    return href

# Function to parse HTML file
def parse_html(file):
    return BeautifulSoup(file, "html.parser")

# Parse paragraphs function
def get_paragraphs(html_soup):
            paragraphs_with_min_words = []
            min_word_count = 8

            for paragraph in html_soup.find_all('p'):
                words = paragraph.text.split()
                if len(words) >= min_word_count:
                    paragraphs_with_min_words.append(paragraph.text.replace('\t',' ').replace('\n',' '))

            return paragraphs_with_min_words

# Get sentences
def get_sentences(paragraph_list):
    sentences = reduce(lambda x,y: x+sent_tokenize(y), [[]]+paragraph_list)
    return sentences

# Streamlit application
def main():
    st.title("DeltaSentiment Application")

    # Upload first HTML file
    st.markdown('<div class="big-font">Please upload your company\'s 10-K report for 2022!</div>', unsafe_allow_html=True)
    uploaded_file1 = st.file_uploader("", key="file1", type="html")

    if uploaded_file1 is not None:
        soup1 = parse_html(uploaded_file1)
        st.success("HTML file uploaded successfully!") # Display the first 1000 characters of the HTML

    # Upload second HTML file
    st.markdown('\n\n')
    st.markdown('<div class="big-font">Please upload your company\'s 10-K report for 2023!</div>', unsafe_allow_html=True)
    uploaded_file2 = st.file_uploader("", key="file2", type="html")

    if uploaded_file2 is not None:
        soup2 = parse_html(uploaded_file2)
        st.success("HTML file uploaded successfully!")  # Display the first 1000 characters of the HTML

    if uploaded_file1!=None and uploaded_file2!=None:
        st.markdown(''' <style>
                    button[kind="primary"] {
                    background-color: #296100;
                    }</style>''', unsafe_allow_html=True)

        if st.button('Preprocess HTML files!', type='primary'):
            st.session_state["button1"] = True
            paragraphs1=get_paragraphs(soup1)
            paragraphs2=get_paragraphs(soup2)
            st.write('Paragraphs parsed from the documents!')
            

            # Add a download button
            download_link = create_download_link({'2022':paragraphs1, '2023':paragraphs2}, filename='paragraphs.json', text_to_show='Click here if you want to download the parsed paragraphs!')

            # Display the download link
            st.markdown(download_link, unsafe_allow_html=True)

            sentences1=get_sentences(paragraphs1)
            sentences2=get_sentences(paragraphs2)


            st.write('Sentences parsed from the documents!')

            # Add a download button
            download_link = create_download_link({'2022':sentences1, '2023':sentences2}, filename='sentences.json', text_to_show='Click here if you want to download the parsed sentences!')

            # Display the download link
            st.markdown(download_link, unsafe_allow_html=True)

            st.session_state['sent1']=sentences1
            st.session_state['sent2']=sentences2

        if st.session_state["button1"]:
            if st.button("Identify relevant sentences!", type='primary'):
                st.session_state["button2"] = True
                dfrel1=pd.DataFrame(st.session_state['sent1'], columns=['text'])
                dfrel2=pd.DataFrame(st.session_state['sent2'], columns=['text'])
                dfrel1['label']=[0]*len(dfrel1)
                dfrel2['label']=[0]*len(dfrel2)

                dfrel1data = CustomDataset(
                texts=dfrel1['text'].tolist(),
                labels=dfrel1['label'].tolist(),
                tokenizer=st.session_state['tokenizer'],
                max_len=256
                )

                dfrel2data = CustomDataset(
                texts=dfrel2['text'].tolist(),
                labels=dfrel2['label'].tolist(),
                tokenizer=st.session_state['tokenizer'],
                max_len=256
                )
                
                dfrel1_loader = DataLoader(dfrel1data, batch_size=512, shuffle=False)
                dfrel2_loader = DataLoader(dfrel2data, batch_size=512, shuffle=False)

                st.session_state['model'].to('cuda')
                st.session_state['model'].eval()

                relsentences1 = predict(st.session_state['model'], dfrel1_loader)
                relsentences2 = predict(st.session_state['model'], dfrel2_loader)
                
                dfrel1['label']=relsentences1
                dfrel2['label']=relsentences2

                dfrel1=dfrel1.sort_values('label', ascending=False)
                dfrel2=dfrel2.sort_values('label', ascending=False)

                dfrel1['label']=dfrel1['label']>=0.9
                dfrel2['label']=dfrel2['label']>=0.9

                dfrel1=dfrel1.query('label==1')
                dfrel2=dfrel2.query('label==1')

                st.write('Our AI identified the following sentences as relevant:')

                st.write('2022')
                st.dataframe(dfrel1)

                st.write('2023')
                st.dataframe(dfrel2)

                st.session_state['dfrel1']=dfrel1
                st.session_state['dfrel2']=dfrel2

        if st.session_state["button2"]:
            if st.button("Calculate DeltaSentiment scores!", type='primary'):
                st.session_state["button3"] = True

                st.session_state['model2'].to('cuda')
                st.session_state['model2'].eval()

                data1 = CustomDataset(
                    texts=st.session_state['dfrel1']['text'].tolist(),
                    labels=st.session_state['dfrel1']['label'].tolist(),
                    tokenizer=st.session_state['tokenizer'],
                    max_len=256
                )
                data2 = CustomDataset(
                    texts=st.session_state['dfrel2']['text'].tolist(),
                    labels=st.session_state['dfrel2']['label'].tolist(),
                    tokenizer=st.session_state['tokenizer'],
                    max_len=256
                )  

                data1_loader = DataLoader(data1, batch_size=512, shuffle=False)
                data2_loader = DataLoader(data2, batch_size=512, shuffle=False)

                sentscores1 = predict2(st.session_state['model2'], data1_loader)
                sentscores2 = predict2(st.session_state['model2'], data2_loader)

                st.session_state['dfrel1']['sentiment_scores']=sentscores1
                st.session_state['dfrel2']['sentiment_scores']=sentscores2

                st.write('The Finbert-Tone model assigned the following sentiment scores to the sentences:')

                st.write('2022')
                st.dataframe(st.session_state['dfrel1'].drop('label', axis=1))

                st.write('2023')
                st.dataframe(st.session_state['dfrel2'].drop('label', axis=1))

                st.write('The sentiment scores of the sentences:')

                dfforplot1=st.session_state['dfrel1'].drop('label', axis=1).copy().sort_values('sentiment_scores')
                dfforplot2=st.session_state['dfrel2'].drop('label', axis=1).copy().sort_values('sentiment_scores')
                
                dfforplot1['sentence_index']=[i for i in range(len(dfforplot1))]
                dfforplot2['sentence_index']=[i for i in range(len(dfforplot2))]

                fig1=px.line(dfforplot1, x='sentence_index', y='sentiment_scores', title='Sentiment Scores of the Sentences for 2022', hover_data='text', labels={'sentiment_scores':'Sentiment Scores', 'sent_index':'Sentence Index'})
                fig2=px.line(dfforplot2, x='sentence_index', y='sentiment_scores', title='Sentiment Scores of the Sentences for 2023', hover_data='text', labels={'sentiment_scores':'Sentiment Scores', 'sent_index':'Sentence Index'})
                fig1.update_layout(title_text='Sentiment Scores of the Sentences for 2022', title_x=0.3625)
                fig2.update_layout(title_text='Sentiment Scores of the Sentences for 2023', title_x=0.3625)

                PLOT_BGCOLOR = "#ebebeb"
                st.markdown(
                            f"""
                            <style>
                            .stPlotlyChart {{
                            outline: 5px solid {PLOT_BGCOLOR};
                            border-radius: 2px;
                            box-shadow: 0 2px 4px 0 rgba(0, 0, 0, 0.05), 0 3px 10px 0 rgba(0, 0, 0, 0.05);
                            }}
                            </style>
                            """, unsafe_allow_html=True
                        )
                st.plotly_chart(fig1)
                st.markdown('\n')
                st.plotly_chart(fig2)


                dfforplot1mean=dfforplot1.sentiment_scores.mean()    
                dfforplot2mean=dfforplot2.sentiment_scores.mean()

                def mean_top_bottom_10_percent(values):
                    n = len(values)
                    top_count = max(int(n * 0.1), 1)
                    bottom_10_percent = values[:top_count]
                    top_10_percent = values[-top_count:]
                    combined_lst=list(bottom_10_percent)+list(top_10_percent)
                    return sum(combined_lst)/len(combined_lst)
                
                dfforplot1meant=mean_top_bottom_10_percent(dfforplot1.sentiment_scores)
                dfforplot2meant=mean_top_bottom_10_percent(dfforplot2.sentiment_scores)

                dfdeltasentimentframe=pd.DataFrame({'values':[dfforplot2mean-dfforplot1mean, dfforplot2meant-dfforplot1meant],
                                                    'names':['DeltaSentiment Score With Mean Scores', 'Delta Sentiment Score With Top/Bottom 10% Mean']})
                
                fig3 = px.bar(dfdeltasentimentframe, x='names', y='values', labels={'names': 'Method', 'values': 'DeltaSentiment Scores'}, title='DeltaSentiment Scores')
                fig3.update_layout(title_text='DeltaSentiment Scores', title_x=0.44)
                st.plotly_chart(fig3)

if __name__ == "__main__":
    main()