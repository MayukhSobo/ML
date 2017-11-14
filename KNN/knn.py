from operator import itemgetter

import numpy as np

from KNN import VERBOSE
from preprocess import Gather
from Utils import euclidean_distance

if VERBOSE:
    from tqdm import tqdm
    from time import sleep


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

    def apply(self, kind='KNN', prediction_mode='absolute', classification=True):
        """
        This is the generic dispatcher function overloaded from
        the Abstract Base Class (Gather) to implement any algorithm
        for Machine Learning Prediction
        Only KNN specific parameter options are supported.
        Prediction mode: absolute only
        Classification: True for now

        :param kind: 'KNN' (constant) -> str
        :param prediction_mode: 'absolute' (constant) -> str
        :param classification: True (constant) -> boolean
        :return: predictions
        """

        # // TODO Regression needs to implemented

        if kind != 'KNN':
            # Dispatcher for KNN should only get kind=KNN
            error_msg = "Dispatcher in KNN got incorrect 'kind'"
            raise AttributeError(error_msg)

        if prediction_mode != 'absolute':
            error_msg = 'Currently KNN only supports absolute prediction mode'
            raise AttributeError(error_msg)

        if not classification:
            raise AttributeError('OOPS!! No support for regression yet!!')

        if KNN.X_train.shape[1] != KNN.X_test.shape[1]:
            error_msg = 'Test and train dataset are not of same dimension'
            raise AttributeError(error_msg)
        if (KNN.y_train.shape[0] != KNN.X_train.shape[0]) or (KNN.y_test.shape[0] != KNN.X_test.shape[0]):
            error_msg = 'Test and/or train samples may not have properly matched labels'
            raise AttributeError(error_msg)

        KNN.fit_transform(self.k)

    @staticmethod
    def _fit_for_point(instance, k):
        """
        Fits the KNN model for one instance.
        This applies the KNN to a particular
        point and returns the predicted class
        label in case of classification.

        :param instance: One test instance -> np.array
        :return: Returns the predicted class
        """

        distances = []
        top_k_neighbours = []
        # KNN.X_train[:, -1] = KNN.y_train
        # print(KNN.X_train)
        for each_train_point in KNN.X_train:
            points = np.array([
                instance,
                each_train_point
            ])

            # Each train point with its distance for test instance
            # each_data_point -> ([1., 2., 3., 4], 7.0)
            data_points = each_train_point, euclidean_distance(points)

            # Storing them so it can be used afterwards
            distances.append(data_points)

            # Sorting data points in increasing order
            # Sort based on the Euclidean distance
            distances.sort(key=itemgetter(1))

            top_k_neighbours = np.array([train_point[0] for train_point in distances[0:k]])

    @staticmethod
    def fit_transform(k):

        # Get all the test instances
        test_instances = KNN.X_test

        # Apply KNN on each instance & get prediction
        if VERBOSE:
            for instance in tqdm(test_instances, ncols=100):
                KNN._fit_for_point(instance=instance, k=k)
                sleep(0.01)
        else:
            for instance in test_instances:
                KNN._fit_for_point(instance=instance, k=k)
