import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import numpy as np
import nengo_dl
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
texts = df['Sentence'].head(500).tolist()
texts = '\n'.join(texts)

# Инициализация токенизатора
maxWordsCount = 5000
tokenizer = Tokenizer(num_words=maxWordsCount)
tokenizer.fit_on_texts([texts])

# Вывод наиболее частотных слов
dist = list(tokenizer.word_counts.items())
sorted_dist = sorted(dist, key=lambda x: x[1], reverse=True)
print("Top 20 most frequent words:")
for word, count in sorted_dist:
    print(f"{word}: {count}")

data = tokenizer.texts_to_sequences([texts])
res = np.array(data[0])
inp_words = 5
n = res.shape[0] - inp_words

X = np.array([res[i:i + inp_words] for i in range(n)])
Y = to_categorical(res[inp_words:], num_classes=maxWordsCount)

X = np.expand_dims(X, axis=1)  # Добавляем ось временных шагов
Y = np.expand_dims(Y, axis=1)  # Добавляем ось временных шагов

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

input_layer = Input(shape=(inp_words,), name='input_layer')

emb_layers = 300
lstm_layers = 150
# Создание модели
embedding_layer = Embedding(maxWordsCount, emb_layers, input_length=inp_words, name='embedding_layer')(input_layer)
gru_layer = LSTM(lstm_layers)(embedding_layer)
output_layer = Dense(maxWordsCount, activation='softmax', name='output_layer')(gru_layer)
model = Model(inputs=input_layer, outputs=output_layer)
model.summary()

converter = nengo_dl.Converter(model)
batch_size = 1
with nengo_dl.Simulator(converter.net, minibatch_size=batch_size) as sim:
    sim.compile(
        optimizer=tf.optimizers.Adam(0.01),
        loss=tf.losses.CategoricalCrossentropy(),
        metrics=tf.metrics.Accuracy()
    )
    def train_model():    
        start_time = time.time()
        history = sim.fit(
            {converter.inputs[input_layer]: X_train},
            {converter.outputs[output_layer]: Y_train},
            validation_data=({converter.inputs[input_layer]: X_test}, {converter.outputs[output_layer]: Y_test}),
            epochs=50
        )
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend(['Train', 'Validation'])
        plt.savefig('train_loss.png')
        sim.save_params(f"./data_words/model_nengo_{batch_size}_{len_text}_{maxWordsCount}_{emb_layers}_{lstm_layers}")

        end_time = time.time()
        duration = end_time - start_time
        print(f"Training completed in {duration:.2f} seconds.")

        return history

    # history = train_model()


    def evaluate_model():
        start_time = time.time()
        results = sim.evaluate(
            {converter.inputs[input_layer]: X_test},
            {converter.outputs[output_layer]: Y_test}
        )
        test_loss = results['loss']
        test_probe_loss = results['probe_loss']
        test_probe_accuracy = results['probe_accuracy']
        print(f"Test loss: {test_loss}")
        print(f"Test probe loss: {test_probe_loss}")
        print(f"Test probe accuracy: {test_probe_accuracy}")
        end_time = time.time()
        duration = end_time - start_time
        print(f"Evaluation completed in {duration:.2f} seconds.")

    # evaluate_model()

# запуск модели 
def run_network(
        params_file=f"data_words/model_nengo_{batch_size}_{len_text}_{maxWordsCount}_{emb_layers}_{lstm_layers}",
        synapse=None,

    ):
    start_time = time.time()
    nengo_converter = nengo_dl.Converter(
        model,
        synapse=synapse,
    )
    
    nengo_input = nengo_converter.inputs[input_layer]
    nengo_output = nengo_converter.outputs[output_layer]
    
    with nengo_dl.Simulator(nengo_converter.net, minibatch_size=1, progress_bar=False) as nengo_sim:
        nengo_sim.load_params(params_file)  
        res = "Telecommunications networks will enable remote"
        data = tokenizer.texts_to_sequences([res])[0]
        for i in range(13):
            x = np.array(data[i: i + inp_words])
            x = np.expand_dims(x, axis=0)  # Добавляем размер батча
            x = np.expand_dims(x, axis=0)  # Добавляем ось временных шагов
            datas = nengo_sim.predict_on_batch({nengo_input: x})
            pred = datas[nengo_output][0]
            indx = pred.argmax(axis=1)[0]
            data.append(indx)
            res += " " + tokenizer.index_word[indx]
        print(res)
    
    end_time = time.time()
    duration = end_time - start_time
    print(f"Testing completed in {duration:.2f} seconds.")

run_network()