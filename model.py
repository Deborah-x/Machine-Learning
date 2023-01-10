import numpy as np
import pickle
import os
from utils.layers import conv_forward_naive, conv_backward_naive, max_pool1d_forward_naive, max_pool1d_backward_naive, affine_forward, affine_backward, dropout_forward, dropout_backward, softmax_loss
from utils.monitor import tic_toc, get_memory
from utils.optim import sgd, sgd_momentum, rmsprop, adam

class TextCNN():
    """
              `--> con_x1 --> pool_x1 --`
              |                         |
    x -> emb_x --> con_x2 --> pool_x2 --|--> fc_x --> out_x --> logit
              |                         |
              `--> con_x3 --> pool_x3 --`
    
    In this network, we accept input embed_x as X, and output logit as Y.

    Parameters size:
    emb-x : (N, M, emd_dim)
    con_x1: (N, dim_channel, M-2, 1)
    pool_x1: (N, dim_channel, 1)
    con_x2: (N, dim_channel, M-3, 1)
    pool_x2: (N, dim_channel, 1)
    con_x3: (N, dim_channel, M-4, 1)
    pool_x3: (N, dim_channel, 1)
    fc_x: (N, dim_channel*3, 1)
    out_x: (N, dim_channel*3)
    logit: (N, num_classes)

    """
    def __init__(self, dim_channel=100, emb_dim=200, num_classes=3, weight_scale=1e-2, dropout_ratio=1.0, reg=0.0, seed=42):
        
        self.reg = reg
        self.seed = np.random.seed() if seed is None else seed
        self.loss_hist = []
        self.acc_hist = []
        self.save_dir = 'model/'

        ### Initialize parameters
        conv_w1 = np.random.normal(loc=0.0, scale=weight_scale, size=(dim_channel, 1, 3, emb_dim))
        conv_b1 = np.zeros(dim_channel)
        conv_w2 = np.random.normal(loc=0.0, scale=weight_scale, size=(dim_channel, 1, 4, emb_dim))
        conv_b2 = np.zeros(dim_channel)
        conv_w3 = np.random.normal(loc=0.0, scale=weight_scale, size=(dim_channel, 1, 5, emb_dim))
        conv_b3 = np.zeros(dim_channel)
        w = np.random.normal(loc=0.0, scale=weight_scale, size=(dim_channel*3, num_classes))
        b = np.zeros(num_classes)

        ### store them in self.params
        self.params = {}
        self.params['conv_w1'] = conv_w1
        self.params['conv_b1'] = conv_b1
        self.params['conv_w2'] = conv_w2
        self.params['conv_b2'] = conv_b2
        self.params['conv_w3'] = conv_w3
        self.params['conv_b3'] = conv_b3
        self.params['w'] = w
        self.params['b'] = b
        self.dropout_param = {'mode':'train', 'p':dropout_ratio, 'seed':self.seed}

        # Initialize config parameters for each module and store them in self.optim_config
        self.optim_config = {'learning_rate': 1e-3}
        self.optim_configs = {}
        for p in self.params:
            self.optim_configs[p] = {}
            for k, v in self.optim_config.items():
                self.optim_configs[p][k] = v 

    @tic_toc
    def train(self, X, Y, optim:str, batch_size=None, epochs=1000, lr=1e-3,  print_every=10, verbose=False):
        optim_dict = {'sgd':sgd, 'sgd_momentum':sgd_momentum, 'rmsprop':rmsprop, 'adam':adam}
        optim = optim_dict.get(optim.lower(), default=adam)
        self.optim_config['learning_rate'] = lr

        for p in self.params:
            self.optim_configs[p] = {}
            for k, v in self.optim_config.items():
                self.optim_configs[p][k] = v

        if batch_size is None:
            batch_size = X.shape[0]

        for i in range(epochs):
            batch_mask = np.random.choice(X.shape[0], batch_size)
            X_batch , Y_batch = X[batch_mask], Y[batch_mask]
            pred_y, caches = self.predict(X_batch)
            loss, dout = softmax_loss(pred_y, Y_batch)
            self.loss_hist.append(loss)
            acc = np.mean(np.argmax(pred_y, Y_batch))
            self.acc_hist.append(acc)
            self.backward(dout, caches, optim, lr)
            if i % print_every == 0:
                print(f"Epoch {i:3d}: loss {loss} accuracy {acc}")
                if verbose:
                    get_memory(verbose=True)


    def predict(self, X, y=None):
        """"
        问题：
        1. max_pool1d_forward_naive和max_pool1d_backward_naive的实现(done)
        2. concatenate的反向传播，split就行了吗？(done)
        3. reshape的反向传播，reshape就行了吗？(done)
        4. 输入怎么搞啊？embedding还没有训练呢。
        首先，从评论中提取review和fit都有的数据，把review作为embedding训练的样本，记住，如果转换的时候出现了新的单词，embedding就是全零。
        然后把review和fit随机分作训练集和验证集，调通这个模型，看看acc。
        """
        
        self.dropout_param['mode'] = 'test' if y is None else 'train'

        X = X.reshape(X.shape[0], 1, X.shape[1], X.shape[2])
        conv_x1, cache1 = conv_forward_naive(X, self.params['conv_w1'], self.params['conv_b1'], self.conv_param)
        conv_x1 = conv_x1.squeeze(-1)
        pool_x1, cache4 = max_pool1d_forward_naive(conv_x1, self.pool_param)

        conv_x2, cache2 = conv_forward_naive(pool_x1, self.params['conv_w2'], self.params['conv_b2'], self.conv_param)
        conv_x2 = conv_x2.squeeze(-1)
        pool_x2, cache5 = max_pool1d_forward_naive(conv_x2, self.pool_param)

        conv_x3, cache3 = conv_forward_naive(pool_x2, self.params['conv_w3'], self.params['conv_b3'], self.conv_param)
        conv_x3 = conv_x3.squeeze(-1)
        pool_x3, cache6 = max_pool1d_forward_naive(conv_x3, self.pool_param)
        
        fc_x = np.concatenate(pool_x1, pool_x2, pool_x3, axis=1)
        fc_x, cache7 = dropout_forward(fc_x, self.dropout_param)
        out_x = fc_x.squeeze(-1)

        logit, cache8 = affine_forward(out_x, self.params['w'], self.params['b'])
        caches = [cache1, cache2, cache3, cache4, cache5, cache6, cache7, cache8]

        if y is not None:
            return logit, caches
        else:
            return logit


    def backward(self, dout, caches, optim):
        cache1, cache2, cache3, cache4, cache5, cache6, cache7, cache8 = caches
        reg, grads = self.reg, {}
        dout, dw, db = affine_backward(dout, cache8)
        grads['w'] = dw + reg * self.params['w'] 
        grads['b'] = db
        dout = dropout_backward(dout, cache7)
        dout = dout.unsqueeze(-1)
        dim_channel = dout.shape[1] // 3
        dout1, dout2, dout3 = dout[:, :dim_channel, :], dout[:, dim_channel:2*dim_channel, :], dout[:, 2*dim_channel:, :]
        dout1 = max_pool1d_backward_naive(dout1, cache4)
        dout2 = max_pool1d_backward_naive(dout2, cache5)
        dout3 = max_pool1d_backward_naive(dout3, cache6)
        dout1 = dout1.unsqueeze(-1)
        dout2 = dout2.unsqueeze(-1)
        dout3 = dout3.unsqueeze(-1)
        dout1, dw1, db1 = conv_backward_naive(dout1, cache1)
        dout2, dw2, db2 = conv_backward_naive(dout2, cache2)
        dout3, dw3, db3 = conv_backward_naive(dout3, cache3)
        grads['conv_w1'] = dw1 + reg * self.params['conv_w1']
        grads['conv_b1'] = db1
        grads['conv_w2'] = dw2 + reg * self.params['conv_w2']
        grads['conv_b2'] = db2
        grads['conv_w3'] = dw3 + reg * self.params['conv_w3']
        grads['conv_b3'] = db3

        for p, w in self.params.item():
            dw = grads[p]
            config = self.optim_configs[p]
            next_w, next_config = optim(w, dw, config)
            self.params[p] = next_w
            self.optim_configs[p] = next_config

        return grads

    def save_model(self, best=False, epoch=None):
        os.mkdir(self.save_path, exist_ok=True)
        if best:
            save_path = os.path.join(self.save_dir, 'best_model.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
        else:
            save_path = os.path.join(self.save_dir, f"epoch{epoch}.pkl")
            with open(save_path, 'wb') as f:
                pickle.dump(self, f)
    
    def load_model(self, path='best_model.pkl'):
        if path.startswith('epoch'):
            print(f"Loading model from {path}")
        else:
            print(f"loading best model from {path}")
        path = os.path.join(self.save_dir, path)
        self = pickle.load(open(path, 'rb'))

    def get_param(self):
        return self.params

    def get_acc(self):
        return self.acc_hist

    def get_loss(self):
        return self.loss_hist      
