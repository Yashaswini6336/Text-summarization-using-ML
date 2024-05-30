# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 19:45:22 2022

@author: YASHASHWINI K
"""
from datetime import datetime
start = datetime.now()
import pandas as pd
import numpy as np
Data = pd.read_csv('C:/Project/wikihow/wikihowAll.csv')
Data = Data.astype(str)
rows, columns = Data.shape
#data pre-processing
df_2 = Data[Data['headline'].isnull() == False]
df_2 = df_2[df_2['headline'] != 'nan']
df_2['headline'] = df_2['headline'].astype(str)
df_2.drop_duplicates(subset=['headline'], inplace=True)
df_2.drop('text', inplace=True, axis=1)
df_2.drop('title', inplace =True, axis=1)
sum2=[]
df = Data[Data['text'].isnull() == False]
df = Data[Data['headline'].isnull() == False]
df = df[df['text'] != 'nan']
df.drop_duplicates(subset=['text'], inplace=True)
df
df.drop('headline', inplace=True, axis=1)
df.drop('title', inplace =True, axis=1)
import sumy
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
import nltk
nltk.download('punkt')
sum1=[]
p1=[]
r1=[]
fm1=[]
pL=[]
rL=[]
fmL=[]
cos=[]
lendf1=[]
lendf2=[]
lenb=[]
for i in range(0,100):
    df1=df.iloc[i,:].str.cat()
    print("text is:")
    print(df1)
    df2=df_2.iloc[i,:].str.cat()
    print("headline is:")
    print(df2)
    parser = PlaintextParser.from_string(df1,Tokenizer("english"))
    from sumy.summarizers.lsa import LsaSummarizer
    lsa_summarizer = LsaSummarizer()
    summary_2 = lsa_summarizer(parser.document,4)
    print("summary using lsa is:")
    for sentence in summary_2:
        print(sentence)
        b=str(summary_2)
    sum1.append(b)
#appling ROUGE to machine generated summary and reference summary
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    print("rouge score for lsa summarizer")
    scores=scorer.score( df2,b)
    scores
    print(scores)
    lendf1.append(len(df1))
    lendf2.append(len(df2))
    lenb.append(len(b))
    p1.append(scores['rouge1'][0])
    r1.append(scores['rouge1'][1])
    fm1.append(scores['rouge1'][2])
    pL.append(scores['rougeL'][0])
    rL.append(scores['rougeL'][1])
    fmL.append(scores['rougeL'][2])
#appling cosine similarity to machine generated summary and reference summary
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    Document1=df2
    Document2=b
    corpus = [Document1,Document2]

    X_train_counts = count_vect.fit_transform(corpus)

    pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names(),index=['Document 1','Document 2'])
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    trsfm=vectorizer.fit_transform(corpus)
    pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['Document 1','Document 2'])

    from sklearn.metrics.pairwise import cosine_similarity

    print(cosine_similarity(trsfm[0:1], trsfm))
    cos.append(cosine_similarity(trsfm[0:1], trsfm))
print(datetime.now() - start)
resultofr1=pd.DataFrame(r1)
resultofr1.to_csv("LSAr1result.csv")
resultofrL=pd.DataFrame(rL)
resultofrL.to_csv("LSArLresult.csv")
resultofp1=pd.DataFrame(p1)
resultofp1.to_csv("LSAp1result.csv")
resultofpL=pd.DataFrame(pL)
resultofpL.to_csv("LSApLresult.csv")
resultoffm1=pd.DataFrame(fm1)
resultoffm1.to_csv("LSAfm1result.csv")
resultoffmL=pd.DataFrame(fmL)
resultoffmL.to_csv("LSAfmLresult.csv")