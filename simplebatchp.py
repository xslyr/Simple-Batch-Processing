#/usr/bin/python3
#!encoding:utf8

import types
import pandas as pd
import numpy as np


class SimpleBatchP:
    model = None
    x = None
    y = None
    length = 0

    def __init__(self, model:types.FunctionType, x, y=None):
        self.model = model
        self.x = x
        self.y = y
        if isinstance(x, pd.DataFrame): self.length = len(x.shape[0])
        elif isinstance(x, np.array): self.length = len(x.shape[0])


    def do_sliced_task(self, blocks_of_execution=2, model_params={}):
        batch_size = int(self.length/blocks_of_execution)
        first_time=True

        for i in range(blocks_of_execution+1):
            try:
                if (i+1)*batch_size > self.length: raise Exception
                if self.y is not None:    
                    self.model(self.x[ i*batch_size:(i+1)*batch_size,:],
                               self.y[ i*batch_size:(i+1)*batch_size,:],
                               **model_params)
                else:
                    self.model(self.x[ i*batch_size:(i+1)*batch_size,:],**model_params)

            except Exception as e:
                if self.y is not None:
                    self.model(self.x[i*batch_size:self.length,:],
                               self.y[i*batch_size:self.length,:],
                               **model_params)
                else:
                    self.model(self.x[i*batch_size:self.length,:],**model_params)


    



    # ESTUDAR MELHORIA:
    #
    # Com C consiguimos manusear melhor a alocação de memória
    # Será possivel medir o tamanho e a porcentagem que a nossa 
    # execução ocupará para que ... de forma iterativa sugerirmos
    # uma quantidade de blocks_of_execution mais conveniente ?
    # 
