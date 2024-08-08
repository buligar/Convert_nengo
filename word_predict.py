import time
import numpy as np
import pandas as pd
from keras.layers import Dense, Input, Embedding, LSTM
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# предсказание слова 

len_text = 500
df = pd.read_csv('data_words/Tense.csv')
texts = df['Sentence'].head(len_text).tolist()
texts = '\n'.join(texts)

# Инициализация токенизатора
maxWordsCount = 5000
tokenizer = Tokenizer(num_words=maxWordsCount)
tokenizer.fit_on_texts([texts])

# Получение частоты слов
dist = list(tokenizer.word_counts.items())
sorted_dist = sorted(dist, key=lambda x: x[1], reverse=True)
print("Топ 20 самых частых слов:")
for word, count in sorted_dist[:20]:
    print(f"{word}: {count}")

data = tokenizer.texts_to_sequences([texts])
res = np.array(data[0])

inp_words = 5
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words] for i in range(n)])
Y = to_categorical(res[inp_words:], num_classes=maxWordsCount)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

input_layer = Input(shape=(inp_words,), name='input_layer')

emb_layers = 300
lstm_layers = 150

# Создание модели
embedding_layer = Embedding(maxWordsCount, emb_layers, input_length=inp_words, name='embedding_layer')(input_layer)
gru_layer = LSTM(lstm_layers)(embedding_layer)
output_layer = Dense(maxWordsCount, activation='softmax', name='output_layer')(gru_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

# Компиляция модели
model.compile(optimizer=tf.optimizers.Adam(0.01),
              loss=tf.losses.CategoricalCrossentropy(),
              metrics=[tf.metrics.Accuracy()])

# Обучение модели
def train_model():
    start_time = time.time()
    batch_size = 64
    history = model.fit(X_train, Y_train,
                        validation_data=(X_test, Y_test),
                        epochs=50,
                        batch_size=batch_size)
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(['Train', 'Validation'])
    plt.savefig('train_loss.png')

    end_time = time.time()
    duration = end_time - start_time
    print(f"Training completed in {duration:.2f} seconds.")

    model.save(f'data_words/model_{batch_size}_{len_text}_{maxWordsCount}_{emb_layers}_{lstm_layers}.h5')
    
    return history

history = train_model()

# Оценка модели
def evaluate_model():
    start_time = time.time()
    results = model.evaluate(X_test, Y_test)
    print(f"Test loss: {results[0]}")
    print(f"Test accuracy: {results[1]}")

    end_time = time.time()
    duration = end_time - start_time
    print(f"Evaluation completed in {duration:.2f} seconds.")

evaluate_model()

# Запуск модели
def run_network():
    start_time = time.time()
    res = "Telecommunications networks will enable remote"
    data = tokenizer.texts_to_sequences([res])[0]
    for i in range(13):
        x = np.array(data[i: i + inp_words])
        x = np.expand_dims(x, axis=0)
        pred = model.predict(x)
        indx = pred.argmax(axis=1)[0]
        data.append(indx)
        res += " " + tokenizer.index_word[indx]  # дописываем слово
    print(res)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Testing completed in {duration:.2f} seconds.")

run_network()
