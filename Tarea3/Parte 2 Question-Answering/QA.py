import time, os, pickle, json
from collections import defaultdict
import numpy as np

from keras.layers import Input,RepeatVector,TimeDistributed,Dense,Embedding,Flatten,Activation,Permute,Lambda
from keras.layers import CuDNNGRU
from keras.models import Model
from keras import backend as K

from keras.utils import plot_model


with open(os.path.join(os.getcwd(),'temp','Xtrain_question.pickle'),'rb') as p_file:
    Xtrain_question = pickle.load(p_file)
with open(os.path.join(os.getcwd(),'temp','Xtest_question.pickle'),'rb') as p_file:
    Xtest_question = pickle.load(p_file)
with open(os.path.join(os.getcwd(),'temp','X_answers.pickle'),'rb') as p_file:
    X_answers = pickle.load(p_file)

input_var = json.load(open(os.path.join(os.getcwd(),'temp','input_var.json'),'rb'))

lenght_output = len(Xtrain_question[0])
hidden_dim = 128
embedding_vector = 64
max_input_lenght = int(input_var['max_input_lenght'])
max_output_lenght = int(input_var['max_output_lenght'])
input_dim = int(input_var['input_dim'])
output_dim = int(input_var['out_dim'])

encoder_input = Input(shape=(max_input_lenght,))
embedded = Embedding(input_dim=input_dim,output_dim=embedding_vector,input_length=max_input_lenght)(encoder_input)
encoder = CuDNNGRU(hidden_dim, return_sequences=True)(embedded)

attention = TimeDistributed(Dense(max_output_lenght, activation='tanh'))(encoder)

attention = Permute([2, 1])(attention)
attention = Activation('softmax')(attention)
attention = Permute([2, 1])(attention)


def attention_multiply(vects):
    encoder, attention = vects
    return K.batch_dot(attention, encoder, axes=1)


sent_representation = Lambda(attention_multiply)([encoder, attention])
decoder = CuDNNGRU(hidden_dim, return_sequences=True)(sent_representation)
#probabilities = TimeDistributed(Dense(len(vocab_answer), activation="softmax"))(decoder)
probabilities = TimeDistributed(Dense(output_dim, activation="softmax"))(decoder)

model = Model(encoder_input,probabilities)
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
model.summary()


if not os.path.exists(os.path.join(os.getcwd(),'results')):
   os.makedirs(os.path.join(os.getcwd(),'results'))

plot_model(model, to_file=os.path.join(os.getcwd(),'results','model_CuDNNGRU.png'))

X_answers = X_answers.reshape(X_answers.shape[0],X_answers.shape[1],1)
#X_answers.shape

model_history = defaultdict(list)
start = time.time()
model_history['history'] = model.fit(Xtrain_question,X_answers,epochs=10,batch_size=64,validation_split=0.2)
end = time.time()-start
model_history['time'] = end
model.save(os.path.join(os.getcwd(),'results','model_CuDNNGRU_.h5'))

with open(os.path.join(os.getcwd(),'results','model_CuDNNGRU_history.pickle'), 'wb') as r_file:
    pickle.dump(model_history['history'].history,r_file)
with open(os.path.join(os.getcwd(),'results','model_CuDNNGRU_time.pickle'), 'wb') as r_file:
    pickle.dump(model_history['time'],r_file)#model.save(os.path.join(os.getcwd(),'results','model_CuDNNGRU_.h5'))

