# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 06:05:26 2022

@author: YASHASHWINI K
"""

import time
start = time.time()
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
lend=[]
for i in range(0,100):
    df1=df.iloc[i,:].str.cat()
    print("text is:")
    print(df1)
    df2=df_2.iloc[i,:].str.cat()
    print("headline is:")
    print(df2)
    parser = PlaintextParser.from_string(df1,Tokenizer("english"))
    from sumy.summarizers.text_rank import TextRankSummarizer
    txtRank = TextRankSummarizer()
    summary_4 = txtRank(parser.document,4)
    print("summary using texrank is:")
    for summary in summary_4:
        print(summary)
        d=str(summary_4)
        sum1.append(d) 
#appling ROUGE to machine generated summary and reference summary
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    print("rouge score for TexRank summarizer")
    scores=scorer.score( df2,d)
    scores
    print(scores)
    lendf1.append(len(df1))
    lendf2.append(len(df2))
    lend.append(len(d))
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
    Document2=d
    corpus = [Document1,Document2]

    X_train_counts = count_vect.fit_transform(corpus)

    pd.DataFrame(X_train_counts.toarray(),columns=count_vect.get_feature_names(),index=['Document 1','Document 2'])
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    trsfm=vectorizer.fit_transform(corpus)
    pd.DataFrame(trsfm.toarray(),columns=vectorizer.get_feature_names(),index=['Document 1','Document 2'])

    from sklearn.metrics.pairwise import cosine_similarity

    cosine_similarity(trsfm[0:1], trsfm)
    cos.append(cosine_similarity(trsfm[0:1], trsfm))
end = time.time()
print(end - start)
textresultofr1=pd.DataFrame(r1)
textresultofr1.to_csv("textr1result.csv")
textresultofrL=pd.DataFrame(rL)
textresultofrL.to_csv("textrLresult.csv")
textresultofp1=pd.DataFrame(p1)
textresultofp1.to_csv("textp1result.csv")
textresultofpL=pd.DataFrame(pL)
textresultofpL.to_csv("textpLresult.csv")
textresultoffm1=pd.DataFrame(fm1)
textresultoffm1.to_csv("textfm1result.csv")
textresultoffmL=pd.DataFrame(fmL)
textresultoffmL.to_csv("textfmLresult.csv")
