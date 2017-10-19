import numpy as np
import pandas as pd
from preprocess import Gather


class KNN(Gather):
    """
    Class to impplement the basic KNN prediction
    on the dataset. It inherits the ```Gather```
    class to collect and process the data an split
    it in testing and traing sections. 
    """
    def __init__(self, path, colnames=None, typ='csv', **kwargs):
        # Use of self.__class__ was giving linter error but it's no big deal
        super(self.__class__, self).__init__(path, colnames, typ)
        label = kwargs.get('label', None)
        n_rounds = kwargs.get('n_rounds', 10)
        copy = kwargs.get('copy', False)
        split_ratio = kwargs.get('split_ratio', 0.4)
        reject_cols = kwargs.get('reject_cols', None)
        numpy_array = kwargs.get('numpy_array', True)
        if label == None:
            raise ValueError('No Label!! Unsupervised problems are Unknown!')
        self.X_train, self.X_test, self.y_train, self.y_test = self.train_test_split(label, 
                                                                                    n_rounds, 
                                                                                    copy, 
                                                                                    split_ratio, 
                                                                                    reject_cols, 
                                                                                    numpy_array)


def main(path, cols=None):
    data_params = {
        'label': 'species'
    }
    knn = KNN(path=path, colnames=cols, **data_params)
    print(knn.X_train.shape)
    print(knn.X_test.shape)
    print(knn.y_train.shape)
    print(knn.y_test.shape)

if __name__ == '__main__':
    pth = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    cols = [
        'sepal_legth',
        'sepal_width',
        'petal_length',
        'petal_width',
        'species'
    ]
    main(pth, cols)
