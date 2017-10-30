from preprocess import Gather
import numpy as np


class KNN(Gather):
    """
    Class to implement the basic KNN prediction
    on the dataset. It inherits the ```Gather```
    class to collect and process the data an split
    it in testing and training sections.

    """

    X_train = None
    X_test = None
    y_train = None
    y_test = None

    def __init__(self, k, path, colnames=None, typ='csv', **kwargs):
        super(self.__class__, self).__init__(path, colnames, typ)
        self.k = k
        label = kwargs.get('label', None)
        n_rounds = kwargs.get('n_rounds', 10)
        copy = kwargs.get('copy', False)
        split_ratio = kwargs.get('split_ratio', 0.4)
        reject_cols = kwargs.get('reject_cols', None)
        numpy_array = kwargs.get('numpy_array', True)
        if not label or label == '':
            raise ValueError('No Label!! Unsupervised problems are Unknown!')
        KNN.X_train, KNN.X_test, KNN.y_train, KNN.y_test = self.train_test_split(label,
                                                                                    n_rounds, 
                                                                                    copy, 
                                                                                    split_ratio, 
                                                                                    reject_cols, 
                                                                                    numpy_array)

    def dispatcher(self, kind=None, prediction_mode='absolute', classification=True):
        """
        This is the generics dispatcher function overloaded from
        the Abstract Base Class (Gather) to implement any algorithm
        for Machine Learning Prediction
        Only KNN specific parameter options are supported.
        Prediction mode: absolute only
        Classification: True always

        :param X_train: Features dataset for training -> np.ndarray
        :param X_test: Features for testing -> np.ndarray
        :param y_train: Labels for training -> np.array
        :param y_test: Labels for testing -> np.array
        :param kind: 'KNN' (constant) -> str
        :param prediction_mode: 'absolute' (constant) -> str
        :param classification: True (constant) -> boolean
        :return: predictions & eval metric information
        """

        if kind != 'KNN':
            # Dispatcher for KNN should only get kind=KNN
            raise AttributeError("Dispatcher in KNN got incorrect 'kind'")

        if prediction_mode != 'absolute':
            raise AttributeError('Currently KNN only supports absolute prediction mode')

        if not classification:
            raise AttributeError('KNN can only be used for Classification')

        if KNN.X_train.shape[1] != KNN.X_test.shape[1]:
            raise AttributeError('Test and train dataset are not of same dimension')

        if (KNN.y_train.shape[0] != KNN.X_train.shape[0]) or (KNN.y_test.shape[0] != KNN.X_test.shape[0]):
            raise AttributeError('Test and/or train samples may not have properly matched labels')

        KNN.fit_transform(self.k)


    @staticmethod
    def fit_transform(k):

        def euclidean_distance(data_points):
            assert len(data_points) == 2
            return np.sqrt(np.sum((data_points[0] - data_points[1]) ** 2))

        oneTestInstance = KNN.X_test[0]
        for each_train_points in KNN.X_train:
            dp = np.array([
                oneTestInstance,
                each_train_points
            ])
            print(euclidean_distance(dp))

