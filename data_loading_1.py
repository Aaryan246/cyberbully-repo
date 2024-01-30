from sklearn.preprocessing import train_test_split
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
import os,sys
from preprocessor_1 import preprocess
from preprocessor_2 import text_preprocessing_pipeline
from gpt_data import returnGPTDataframe
def  returnData():
    data_CCD = pd.read_csv("./data/CyberBullying Comments Dataset.csv")
    data_c_t = pd.read_csv("./data/agr_en_train.csv")
    data_c_d = pd.read_csv("./data/agr_en_dev.csv")
    data_cc_ct = pd.read_csv("./data/cyberbullying_tweets.csv")
    data_c_t = data_c_t.rename(columns = {"Well said sonu..you have courage to stand against dadagiri of Muslims" : "text","OAG" : "label"})
    data_c_t = data_c_t.drop(["facebook_corpus_msr_1723796"],axis = 1)
    data_c_d = data_c_d.rename(columns = {"The quality of re made now makes me think it is something to be bought from fish market" : "text","CAG" : "label"})
    data_c_d = data_c_d.drop(["facebook_corpus_msr_451811"],axis = 1)
    data_cc_ct = data_cc_ct.rename(columns = {"tweet_text" : "text","cyberbullying_type" : "label"})
    data_CCD = data_CCD.rename(columns = {"Text" : "text","CB_Label" : "label"})
    data_c_t['label'] = data_c_t['label'].replace({'OAG': 1, 'CAG': 1, 'NAG': 0})
    data_c_d['label'] = data_c_d['label'].replace({'OAG': 1, 'CAG': 1, 'NAG': 0})
    data_cc_ct['label'] = data_cc_ct['label'].replace({'cyberbullying': 1, 'not_cyberbullying': 0,'gender' : 1,'religion':1,'other_cyberbullying' : 1,'age' : 1,'ethnicity' : 1})
    data_joined = pd.concat([data_c_t,data_c_d,data_cc_ct,data_CCD],ignore_index=True)
    gpt_data = returnGPTDataframe()
    data_joined["text"] = data_joined["text"].apply(lambda x : preprocess(x))
    data_joined = data_joined[data_joined['text'].apply(lambda x: len(x) > 0)]
    gpt_data["text"] = gpt_data["text"].apply(lambda x : preprocess(x))
    gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]
    data_joined["text"] = data_joined["text"].apply(lambda x : text_preprocessing_pipeline(x))
    data_joined = data_joined[data_joined['text'].apply(lambda x: len(x) > 0)]
    gpt_data["text"] = gpt_data["text"].apply(lambda x : text_preprocessing_pipeline(x))
    gpt_data = gpt_data[gpt_data['text'].apply(lambda x: len(x) > 0)]
    train_df, valid_df = train_test_split(data_joined, test_size=0.2, stratify=data_joined['label'], random_state=42)
    train_df = pd.concat([train_df, gpt_data], ignore_index=True)
    train_df = train_df.reset_index()
    valid_df = valid_df.reset_index()
    train_df = train_df.drop(['index'],axis=1)
    valid_df = valid_df.drop(['index'],axis=1)
    np.random.seed(41)
    train_shuffled = train_df.reindex(np.random.permutation(train_df.index))
    valid_shuffled = valid_df.reindex(np.random.permutation(valid_df.index))
    not_cyber = train_shuffled[train_shuffled['label'] == 0]
    cyber = train_shuffled[train_shuffled['label'] == 1]
    concated_train = pd.concat([not_cyber,cyber], ignore_index=True)
    concated_train['LABEL'] = 0
    concated_train.loc[concated_train['label'] == 0, 'LABEL'] = 0
    concated_train.loc[concated_train['label'] == 1, 'LABEL'] = 1
    not_cyber = valid_shuffled[valid_shuffled['label'] == 0]
    cyber = valid_shuffled[valid_shuffled['label'] == 1]
    concated_valid = pd.concat([not_cyber,cyber], ignore_index=True)
    concated_valid['LABEL'] = 0
    concated_valid.loc[concated_valid['label'] == 0, 'LABEL'] = 0
    concated_valid.loc[concated_valid['label'] == 1, 'LABEL'] = 1
    X_train = concated_train['text']
    X_valid = concated_valid['text']
    in_features = len(list(X_train))
    class_list = [0,1]
    class_num = len(class_list)
    y_train = to_categorical(concated_train['LABEL'], num_classes=2)
    y_valid = to_categorical(concated_valid['LABEL'], num_classes=2)