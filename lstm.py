
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Bidirectional, LSTM 
import matplotlib.pyplot as plt
gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


STOPWORDS = set(stopwords.words('english'))

print(tf.__version__)

vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

articles = []
labels = []

with open("bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter = ',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace( token , ' ')
            article = article.replace(' ', ' ')
        articles.append(article)

    # print(articles[0])
# print(len(labels))
# print(len(articles))

train_size = int(len(articles)*training_portion)

train_articles = articles[0:train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]

# print(train_size)
# print(len(train_articles))
# print(len(train_labels))
# print(len(validation_articles))
# print(len(validation_labels))

tokenizer = Tokenizer(num_words = vocab_size, oov_token = oov_tok )
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index
# print(dict(list(word_index1.items())[0:10]))

train_sequences = tokenizer.texts_to_sequences(train_articles)
validation_sequences = tokenizer.texts_to_sequences(validation_articles)
# print(train_sequences[10])

train_padded = pad_sequences(train_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type )
validation_padded = pad_sequences(validation_sequences, maxlen = max_length, padding = padding_type, truncating = trunc_type )

# print(len(train_sequences[0]))
# print(len(train_padded[0]))

# print(len(train_sequences[1]))
# print(len(train_padded[1]))

# print(train_padded[10])
# print(train_padded[0])

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

# print(label_index)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

# print(training_label_seq[0])
# print(training_label_seq[1])
# print(training_label_seq[2])
# print(training_label_seq.shape)

# print(validation_label_seq[0])
# print(validation_label_seq[1])
# print(validation_label_seq[2])
# print(validation_label_seq.shape)

# print(train_padded.shape)
# print(train_padded.shape[-1])

x = Input( shape = (train_padded.shape[-1]))
m = Embedding(vocab_size, embedding_dim)(x)
m = Bidirectional(LSTM(embedding_dim))(m)
m = Dense(embedding_dim, activation = 'relu')(m)
m = Dense(6, activation = 'softmax')(m)

model = Model(inputs = x, outputs = m)

print(model.summary())


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10
history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)


def plot_graphs(history, string1, string2):
    fig = plt.figure()
    a = fig.add_subplot(211)
    
    a.set_xlabel("Epochs")
    a.set_ylabel(string1)
    a.plot(history.history[string1])

    b = fig.add_subplot(212)
    
    b.set_xlabel("Epochs")
    b.set_ylabel(string2)
    b.plot(history.history['val_'+string2])    

    plt.savefig("Figure"+ ".png")
   
plot_graphs(history, "accuracy", "loss")
