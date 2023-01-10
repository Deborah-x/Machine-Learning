from utils.vocab import get_embedding, preprocess
from model import TextCNN
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # We first need to get embedding_dict, which is trained on 2,100 
    # (this number is limited by poor memory capicity of laptop) reviews.
    embedding_dict = get_embedding(path='model/emb.pkl')
    # Then we load the data and split them in training dataset and validation dataset
    (x_train, y_train), (x_test, y_test) = preprocess(embed_dict=embedding_dict)
    # We can see the shape of the data
    # print(f"x_train.shape: {x_train.shape}")
    # print(f"y_train.shape: {y_train.shape}")
    model = TextCNN(dim_channel=100, emb_dim=100, num_classes=3, dropout_ratio=0.5)
    model.train(x_train, y_train, optim='adam', batch_size=512, epochs=10, lr=1e-3, print_every=1, verbose=True)
    model.save_model(epoch=10)   