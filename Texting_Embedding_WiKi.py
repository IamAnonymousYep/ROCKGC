# -*- coding: utf-8 -*-
import sys, os, re
import numpy as np
import pickle
import json
from nltk.corpus import stopwords
import string
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from Texting_Embedding_NELL import *

def generate_matrix_(WordMatrix, corpus, corpus_tfidf, rela2id, rela_list, w_dim=300):
    doc_Matrix = np.zeros(shape=(539, w_dim), dtype='float32')
    for rela in rela_list:
        if rela not in rela2id:
            continue
        id = rela2id[rela]
        index = rela_list.index(rela)
        doc = corpus[index]
        doc_tfidf = corpus_tfidf[index]
        tmp_vec = np.zeros(shape=(w_dim,), dtype='float32')
        non_repeat = list()
        for i in range(len(doc)):
            # print '1', WordMatrix[doc[i]]
            # print doc_tfidf[i]
            # print '2', WordMatrix[doc[i]] * doc_tfidf[i]
            if doc[i] not in non_repeat:
                non_repeat.append(doc[i])
                tmp_vec += WordMatrix[doc[i]] * doc_tfidf[i]
        print(f"id:{id}, len of rela_list: {len(rela_list)}")
        print(rela_list[id], len(non_repeat), len(doc))
        tmp_vec = tmp_vec / max(float(len(non_repeat)),1)
        doc_Matrix[id] = tmp_vec
        # break

    return doc_Matrix

def calculate_tfidf_wiki(rela_list, corpus, word2id):
    tfidf_vec = TfidfVectorizer(stop_words=stopwords.words('english'))
    #transformer=TfidfTransformer(stop_words=stopwords.words('english'))
    tfidf = tfidf_vec.fit_transform(corpus)
    word = tfidf_vec.get_feature_names_out() # list, num of words  initial euquals get_feature_names()
    weight = tfidf.toarray() # (181, num of words)
    weight = weight.astype('float32')

    corpus_tfidf = list()
    corpus_new = list()
    for num in range(len(rela_list)):
        word2tfidf = zip(word, list(weight[num]))
        word2tfidf = dict(word2tfidf)
        assert len(word) == len(list(weight[num]))
        doc_tfidf = list()
        doc_ids = list()
        word_list = corpus[num].split()
        for w in word_list:
            if w in word:
                # if word2tfidf[w] < 0.05:
                #    continue
                doc_tfidf.append(word2tfidf[w])
                doc_ids.append(word2id[w])
                #if rela_list[num] == 'concept:athletebeatathlete':
                    #print '  (', w, word2tfidf[w], ')  ',
        corpus_tfidf.append(doc_tfidf)
        corpus_new.append(doc_ids)

    assert len(corpus_tfidf) == len(rela_list)

    return corpus_tfidf, word, corpus_new

def WiKi_text_embedding(data_path, dataname = 'WiKi'):
    # read rela_document.txt
    rela2id_Wiki = json.load(open(data_path + "/relation2ids"))
    rela2doc = dict()
    with open(data_path + "/rela_document.txt",encoding='utf-8') as f_doc:
        lines = f_doc.readlines()
        for num in range(575):
            rela = lines[7*num].strip().split('###')[0].strip()
            description = lines[7 * num + 1].strip().split('###')[1].strip()
            description = clean_str(description)
            label = lines[7 * num + 2].strip().split('###')[1].strip()
            label = clean_str(label)
            p31 = lines[7 * num + 3].strip().split('###')[1].strip()
            p31 = clean_str(p31)
            p1629 = lines[7 * num + 4].strip().split('###')[1].strip()
            p1629 = clean_str(p1629)
            p1855 = lines[7 * num + 5].strip().split('###')[1].strip()
            p1855 = clean_str(p1855)
            rela2doc[rela] = description +  " " +  label + " " + description +  " " +  description +  " " +  description +  " " + p1629
    vocab = get_vocabulary(rela2doc)
    print('NELL description vocab size %d' % (len(vocab)))
    # word2id, WordMatrix = load_wordembedding_300(data_path, vocab, dataname='WiKi')
    WordMatrix = np.load('./origin_data/Wiki/WordMatrix_300_Wiki.npz')['arr_0']
    word2id = pickle.load(open('./origin_data/Wiki/word2id_300_Wiki.pkl','rb'))
    rela_list, corpus_text = clean_OOV(rela2doc, word2id)
    reldes2ids = dict()
    for i,rela in enumerate(rela_list):
        reldes2ids[rela] = int(i)
    json.dump(reldes2ids, open(data_path + "/reldes2ids", 'w'))
    corpus_tfidf, vocab_tfidf, corpus = calculate_tfidf_wiki(rela_list, corpus_text, word2id)
    rela_matrix_Wiki = generate_matrix_(WordMatrix, corpus, corpus_tfidf, rela2id_Wiki, rela_list)
    rela_matrix_Wiki = rela_matrix_Wiki * 10
    # np.savez(data_path + "/rela_matrix", relaM=rela_matrix_Wiki)
    print("rela_matrix shape %s" % (str(rela_matrix_Wiki.shape)))
