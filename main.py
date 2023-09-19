# BENANE Balignissa Master2  DATA SCIENCE

import pandas as pd
# Charger le fichier CSV
data = pd.read_csv('multilabel-classification-datase.csv',encoding='UTF-8')
print(data.info())
print(data.head())

# Vérifier les valeurs nulles dans chaque colonne
null_counts = data.isnull().sum()
print(null_counts)

# Supprimer les lignes avec des valeurs nulles
data_cleaned = data.dropna()


import numpy as np
seq_len=120
num_samples = len(data)
print(num_samples)


# initialisation des  inputs ids et attention mask en array
xids = np.zeros((num_samples,seq_len))
xmasks = np.zeros((num_samples,seq_len))

print(xids)

# choisir le type de token et l'initialiser
#!pip install git+https://github.com/huggingface/transformers
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

# Tokenizer les phrases
for i, phrase in enumerate(data['ABSTRACT']):
    tokens = tokenizer.encode_plus(phrase, max_length=seq_len, truncation=True,
    padding = 'max_length', add_special_token=True, retur_tensor='tf')

xids[i, :] = tokens['input_ids']
xmasks[i, :] = tokens['attention_mask']

print(xids)
print(xmasks)
labels = np.zeros((num_samples,6))
labels

import tensorflow as tf
dataset = tf.data.Dataset.from_tensor_slices((xids, xmasks, labels))
dataset.take(1) #prendre un seul exemple parmi plusieurs

#Vérifier le nombre de classes
labels[0, :].shape

def map_fc(input_ids, masks, labels):
    return{'input_ids':input_ids, 'attention_mask':masks}, labels
dataset = dataset.map(map_fc)
dataset.take(1)

batch_size = 16
dataset = dataset.shuffle(10000).batch(batch_size, drop_remainder=True)
dataset.take(1)
# Faire le split et vérifier les sizes
split = 0.8
size = int((num_samples/batch_size)*split)
print(size)