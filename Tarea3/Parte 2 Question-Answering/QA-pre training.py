import pandas as pd
import time, os
from collections import defaultdict
import pickle, json


df_train = pd.read_csv(os.path.join(os.getcwd(),'data','train_Q-A.csv'))
df_train.dropna(inplace=True)
df_test = pd.read_csv(os.path.join(os.getcwd(),'data','test_Q.csv'))
df_train.head()
df_train.shape

print("Existen un total de  {} preguntas".format(len(df_train['question'])))
print("Y un total de {} respuestas".format(len(df_train['answer'])))


print("En cuanto a las preguntas para predecir se tiene que son un total de {}".format(
   len(df_test['question']) ))

from nltk.tokenize import word_tokenize

train_questions = [word_tokenize(sentence.lower()) for sentence in df_train["question"]]
test_questions = [word_tokenize(sentence.lower()) for sentence in df_test["question"]]
train_answers = [word_tokenize(sentence) for sentence in df_train["answer"]]

vocab_answer = set()
for sentence in train_answers:
    for word in sentence:
        vocab_answer.add(word)
vocab_answer = ["#end"]+ list(vocab_answer)
print('posibles palabras para respuestas: {}'.format(len(vocab_answer)))
vocabA_indices = {c: i for i, c in enumerate(vocab_answer)}
indices_vocabA = {i: c for i, c in enumerate(vocab_answer)}

vocab_question = set()
for sentence in train_questions+test_questions:
    for word in sentence:
        vocab_question.add(word)
vocab_question = list(vocab_question)
print('posibles palabras para preguntas: {}'.format(len(vocab_question)))
vocabQ_indices = {c: i for i, c in enumerate(vocab_question)}

X_answers = [[vocabA_indices[palabra] for palabra in sentence] for sentence in train_answers]
Xtrain_question = [[vocabQ_indices[palabra] for palabra in sentence] for sentence in train_questions]
Xtest_question = [[vocabQ_indices[palabra] for palabra in sentence] for sentence in test_questions]

import numpy as np
max_input_lenght = np.max(list(map(len,train_questions)))
max_output_lenght = np.max(list(map(len,train_answers)))+1

from keras.preprocessing import sequence

Xtrain_question = sequence.pad_sequences(Xtrain_question,maxlen=max_input_lenght,padding='post',value=0)
Xtest_question = sequence.pad_sequences(Xtest_question,maxlen=max_input_lenght,padding='post',value=0)
X_answers = sequence.pad_sequences(X_answers,maxlen=max_output_lenght,padding='post',value=vocabA_indices["#end"])

#Guardar
if not os.path.exists(os.path.join(os.getcwd(),'temp')):
   os.makedirs(os.path.join(os.getcwd(),'temp'))
pickle.dump(Xtrain_question, open(os.path.join(os.getcwd(),'temp','Xtrain_question.pickle'), 'wb'))
pickle.dump(vocab_question, open(os.path.join(os.getcwd(),'temp','vocab_question.pickle'), 'wb'))
pickle.dump(vocabQ_indices, open(os.path.join(os.getcwd(),'temp','vocabQ_indices.pickle'), 'wb'))
pickle.dump(vocab_answer, open(os.path.join(os.getcwd(),'temp','vocab_answer.pickle'), 'wb'))
pickle.dump(vocabA_indices, open(os.path.join(os.getcwd(),'temp','vocabA_indices.pickle'), 'wb'))
pickle.dump(indices_vocabA, open(os.path.join(os.getcwd(),'temp','indices_vocabA.pickle'), 'wb'))
pickle.dump(Xtest_question, open(os.path.join(os.getcwd(),'temp','Xtest_question.pickle'), 'wb'))
pickle.dump(X_answers, open(os.path.join(os.getcwd(),'temp','X_answers.pickle'), 'wb'))
input_var = defaultdict(int)
input_var['max_input_lenght'] = int(max_input_lenght)
input_var['max_output_lenght'] = int(max_output_lenght)
input_var['input_dim'] = int(len(vocabQ_indices))
input_var['out_dim'] = int(len(vocab_answer))

json.dump(input_var, open(os.path.join(os.getcwd(),'temp','input_var.json'), 'w'))