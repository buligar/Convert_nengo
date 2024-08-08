import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
import nengo_dl
import tensorflow as tf
from keras.layers import Dense, GRU, Input, Bidirectional
from keras.models import Model

# достраивание сигмоиды по трем предыдущей и последующей наблюдениям (двунаправленная рекуррентная сеть)

N = 1000
dataset = np.array([np.sin(x/20) for x in range(N)]) + 0.1*np.random.randn(N)
off = 3
length = off*2+1
X = np.array([ np.diag(np.hstack((dataset[i:i+off], dataset[i+off+1:i+length]))) for i in range(N-length)])
Y = dataset[off:N-off-1]

input_dot = Input((length-1, length-1))
bidire = Bidirectional(GRU(2))(input_dot)
output_bi = Dense(1, activation='linear')(bidire)
model =  Model(input_dot, output_bi)

GRU_model = nengo_dl.Converter(model)

X = X.reshape(993, 1, 36)
Y = np.expand_dims(Y, axis=-1)
Y = np.expand_dims(Y, axis=1)  # добавьте измерение для n_steps

print("X",X.shape) 
print("Y",Y.shape)

def train_model():    
    with nengo_dl.Simulator(GRU_model.net, minibatch_size=1) as sim:
        sim.compile(
            optimizer=tf.optimizers.Adam(0.01),
            loss={GRU_model.outputs[model.output]: tf.losses.MeanSquaredError()},
            metrics={GRU_model.outputs[model.output]: tf.metrics.MeanSquaredError()}
        )
        history = sim.fit(
            {GRU_model.inputs[input_dot]: X},
            {GRU_model.outputs[model.output]: Y},
            epochs=10
        )
        plt.plot(history.history['loss'])
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.show()
        sim.save_params("./data_words/bidirect_snn")
    return history

# history = train_model()


# запуск модели 
def run_network(
        params_file="./data_words/bidirect_snn",
        synapse=None,

    ):
    nengo_converter = nengo_dl.Converter(
        model,
        synapse=synapse,
    )
    
    nengo_input = nengo_converter.inputs[input_dot]
    nengo_output = nengo_converter.outputs[model.output]
    
    with nengo_dl.Simulator(nengo_converter.net, minibatch_size=1, progress_bar=False) as nengo_sim:
        nengo_sim.load_params(params_file)  
        M = 300
        XX = np.zeros(M)
        XX[:off] = dataset[:off]
        for i in range(M-off-1):
            time = np.arange(i,i+3)
            x = np.diag(np.hstack((XX[i:i+off], dataset[i+off+1:i+length])))
            x = np.expand_dims(x, axis=0)
            x = x.reshape(1, 1, 36)
            data = nengo_sim.predict({nengo_input: x})
            XX[i+off] = data[nengo_output][0]

    plt.figure()
    plt.plot(XX[:M])
    plt.plot(dataset[:M])
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

run_network()