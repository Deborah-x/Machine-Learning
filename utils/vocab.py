import pandas as pd
import numpy as np
import re
from tqdm import tqdm
from pprint import pprint
import json
import matplotlib.pyplot as plt
try:
    from keras.models import Input, Model
    from keras.layers import Dense
except:
    pass
try:
    from .layers import *
    from .layer_utils import *
    from .optim import *
    from .monitor import *
except:
    print("Relative import failed. Trying absolute import.")
    from layers import *
    from layer_utils import *
    from optim import *
    from monitor import *

def clean_text( text:str )->str:
    """
    Input: a string
    Output: a cleaned list of words
    """
    # Clean text
    punctuations = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
    stop_words = ['a', 'the', 'is', 'and', 'be', ]
    # Cleaning the urls
    text = re.sub(r"http?s://\S+|www\.\S+", "", text)

    # Cleaning the html elements
    text = re.sub(r"<.*?>", "", text)

    # Removing the punctuations
    for x in text.lower():
        if x in punctuations:
            text = text.replace(x, "")

    # Converting words to lower case
    text = text.lower()

    # Removing stop words
    text = " ".join([word for word in text.split() if word not in stop_words])

    # Cleaning the whitespaces
    text = re.sub(r"\s+", " ", text).strip()

    text = text.split()
    
    return text

def create_context_dictionary(texts):
    """
    Input: a list of string
    Output: a dictionary of context
    """
    # Defining the window for context, in this case (focus word, context word)
    window = 2
    # Creating a placeholder for the scanning of the word list
    word_lists = []
    all_text = []
    for text in texts:

        # Cleaning the text
        text = clean_text(text)
        
        # Appending to the all text list
        all_text += text

        # Creating a context dictionary
        for i, word in enumerate(text):
            for w in range(window):
                # Getting the context that is ahead by *window* words
                if i + 1 + w < len(text): 
                    word_lists.append([word] + [text[(i + 1 + w)]])
                # Getting the context that is behind by *window* words    
                if i - w - 1 >= 0:
                    word_lists.append([word] + [text[(i - w - 1)]])

    # print(all_text)
    return all_text, word_lists

def create_unique_word_dict(text:list) -> dict:
    """
    A method that creates a dictionary where the keys are unique words
    and key values are indices
    """
    # Getting all the unique words from our text and sorting them alphabetically
    words = list(set(text))
    words.sort()

    # Creating the dictionary for the unique words
    unique_word_dict = {}
    for i, word in enumerate(words):
        unique_word_dict.update({
            word: i
        })

    return unique_word_dict

def create_one_hot(unique_word_dict):
    """
    A method that creates a one hot encoding for the unique words
    """
    # Defining the number of features (unique words)
    n_words = len(unique_word_dict)

    # Getting all the unique words 
    words = list(unique_word_dict.keys())

    # Creating the X and Y matrices using one hot encoding
    X = []
    Y = []

    for i, word_list in tqdm(enumerate(word_lists)):
        # Getting the indices
        main_word_index = unique_word_dict.get(word_list[0])
        context_word_index = unique_word_dict.get(word_list[1])

        # Creating the placeholders   
        X_row = np.zeros(n_words)
        Y_row = np.zeros(n_words)

        # One hot encoding the main word
        X_row[main_word_index] = 1

        # One hot encoding the Y matrix words 
        Y_row[context_word_index] = 1

        # Appending to the main matrices
        X.append(X_row)
        Y.append(Y_row)

    # Converting to numpy arrays
    X = np.array(X, dtype=np.int32)
    Y = np.array(Y, dtype=np.int32)

    return words, X, Y
        
class manualNN():
    def __init__(
        self, 
        input_dim, 
        hidden_dims, 
        num_classes, 
        weight_scale=1e-2, 
        learning_rate=1e-3,
        reg=0.0,
        dtype=np.float32
    ) -> None:
        self.num_layers = len(hidden_dims) + 1
        self.reg = reg
        self.params = {}
        """
        Network Structure(normalization, dropout, regularization are not implemented):
        affine  -> affine -> softmax
        
        hidden_dims: list of integers giving the size of each hidden layer.
        """
        
        # Initialize weights and biases, store them in self.params
        layer_dims = [input_dim] + hidden_dims + [num_classes]
        for i in range(len(layer_dims) - 1):
            layer_input_dim = layer_dims[i]
            layer_output_dim = layer_dims[i + 1]
            self.params[f"W{i+1}"] = np.random.normal(loc=0.0, scale=weight_scale
                , size=(layer_input_dim, layer_output_dim))
            print(f"W{i+1}: {self.params[f'W{i+1}']}")
            self.params[f"b{i+1}"] = np.zeros(layer_output_dim)

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

        # Initialize config parameters for each module and store them in self.optim_config
        self.optim_config = {}
        self.optim_configs = {}
        self.optim_config["learning_rate"] = learning_rate
        for p in self.params:
            self.optim_configs[p] = {}
            for k, v in self.optim_config.items():
                self.optim_configs[p][k] = v

        # store loss and accuracy history
        self.loss_hist = []
        self.acc_hist = []
    
    @tic_toc
    def train(self, X, y, optim : str, batch_size=None, epochs=100, lr=1e-3, print_every=10, verbose=False):

        optim_dict = {"sgd":sgd, "sgd_momentum":sgd_momentum, "rmsprop":rmsprop, "adam":adam}
        optim = optim_dict[optim]
        # only learning rate is needed for now, but learning rate decay will be added later
        self.optim_config["learning_rate"] = lr
        for p in self.params:
            self.optim_configs[p] = {}
            for k, v in self.optim_config.items():
                self.optim_configs[p][k] = v
        
        if batch_size is None:
            batch_size = X.shape[0]
        
        for i in range(epochs):
            batch_mask = np.random.choice(X.shape[0], batch_size)
            X_batch, y_batch = X[batch_mask], y[batch_mask]
            pred_y, caches = self.predict(X_batch, y_batch)
            loss, dout = softmax_loss(pred_y, y_batch)
            self.loss_hist.append(loss)
            acc = np.mean(np.argmax(pred_y, axis=1) == y_batch)
            self.acc_hist.append(acc)
            self.backward(dout, caches, optim, lr)
            if i % print_every == 0:
                print(f"Epoch {i}: loss {loss} accuracy {acc}")
                if verbose:
                    get_memory(verbose=True)
        

    
    def predict(self, X, y=None):
        """
        This function implement a forward pass of the network.
        If you pass y, it will trate it as a training process.
        If you don't pass y, it will trate it as a testing process.
        """
        mode = "test" if y is None else "train"

        caches = []
        out = X
        for i in range(1, self.num_layers):
            W = self.params[f"W{i}"]
            b = self.params[f"b{i}"]
            out, cache = affine_forward(out, W, b)
            caches.append(cache)

        W = self.params[f"W{self.num_layers}"]
        b = self.params[f"b{self.num_layers}"]
        out, cache = affine_forward(out, W, b)
        caches.append(cache)
        scores = out
        # get_memory(verbose=True)

        return scores, caches

    
    def backward(self, dout, caches, optim, lr):
        reg, grads = self.reg, {}
        
        # Backward pass: compute gradients
        dout, dw, db = affine_backward(dout, caches[-1])
        grads[f"dW{self.num_layers}"] = dw
        grads[f"db{self.num_layers}"] = db


        for i in range(self.num_layers - 2, -1, -1):
            cache = caches[i]
            dout, dw, db = affine_backward(dout, cache)
            dw += reg * self.params[f"W{i+1}"]
            grads[f"dW{i+1}"] = dw
            grads[f"db{i+1}"] = db

        # Update parameters
        for p, w in self.params.items():
            dw = grads[f"d{p}"]
            config = self.optim_configs[p]
            next_w, next_config = optim(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config

        return grads

    def save_model(self, path="model.json", epoch=None):
        save_path = f"epoch_{epoch}_{path}"
        with open(save_path, "w") as f:
            json.dump(self.params, f)

    def load_model(self, path="model.json"):
        if path.startswith("epoch_"):
            epoch = int(path.split("_")[1])
            print(f"Loading model which has been trained {epoch} epoches...")
        with open(path, "r") as f:
            self.params = json.load(f)

    def get_param(self):
        return self.params

def test():
    X = np.random.randn(1000, 10)
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    W = np.array([[0.1, 0.2], [0.2, 0.1]]).repeat(5, axis=0)
    y = np.dot(X, W) + np.random.random((1000, 2)) * 0.1
    y = y.argmax(axis=1).reshape(-1, 1)
    model = manualNN(X.shape[1], [], 2, reg=1, weight_scale=0.2)
    model.train(X, y, optim='adam', epochs=1000, lr=0.01, batch_size=256)
    print(model.get_param()['W1'])
    print(model.get_param()['b1'])


if __name__ == '__main__':
    texts = pd.read_csv('sample.csv')
    texts = [x for x in texts['text']]
    all_text, word_lists = create_context_dictionary(texts)
    unique_word_dict = create_unique_word_dict(all_text)
    words, X, Y = create_one_hot(unique_word_dict)
    x, y = X, Y.argmax(axis=1).reshape(-1, 1)
    embed_size = 2

    ### implement your own neural network here
    model = manualNN(X.shape[1], [embed_size], Y.shape[1], weight_scale=1)
    model.train(x, y, optim='adam', epochs=1000, lr=0.001, batch_size=256)
    embedding_dict = {}
    weights = model.get_param()['W1']
    
    # Creating a dictionary to store the embeddings in. The key is a unique word and 
    # the value is the numeric vector
    embedding_dict = {}
    for word in words: 
        embedding_dict.update({
            word: weights[unique_word_dict.get(word)]
            })
    pprint(embedding_dict)
    
    plt.figure(figsize=(10, 10))
    for word in list(unique_word_dict.keys()):
        coord = embedding_dict.get(word)
        plt.scatter(coord[0], coord[1])
        plt.annotate(word, (coord[0], coord[1]))
    plt.show()


    # ### using the library
    # # Defining the neural network
    # inp = Input(shape=(X.shape[1],))
    # x = Dense(units=embed_size, activation='linear')(inp)
    # x = Dense(units=Y.shape[1], activation='softmax')(x)
    # model = Model(inputs=inp, outputs=x)
    # model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])

    # # Optimizing the network weights
    # model.fit(
    #     x=X, 
    #     y=Y, 
    #     batch_size=256,
    #     epochs=1000
    #     )
    # y_pred = model.predict(X[:10])
    # print(f"Predictions: {y_pred}")
    # # Obtaining the weights from the neural network. 
    # # These are the so called word embeddings

    # # The input layer 
    # weights = model.get_weights()[0]

    # # Creating a dictionary to store the embeddings in. The key is a unique word and 
    # # the value is the numeric vector
    # embedding_dict = {}
    # for word in words: 
    #     embedding_dict.update({
    #         word: weights[unique_word_dict.get(word)]
    #     })
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 10))
    # for word in list(unique_word_dict.keys()):
    #     coord = embedding_dict.get(word)
    #     plt.scatter(coord[0], coord[1])
    #     plt.annotate(word, (coord[0], coord[1]))
    # plt.show()
    
