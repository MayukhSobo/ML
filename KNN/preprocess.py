import pandas as pd # For all data processing
import numpy as np
from math import ceil
from warnings import warn

class Gather(object):
    """
    Gather module collects all the
    data passed through its constructor
    and performs the following opertations.
        1. Head (n = 5)
        2. Tail (n = 5)
        3. Size (Rows and Columns)
        4. Shuffle the dataset (n_rounds times)
        5. Split the dataset in 2 parts (train, cross-validation)
        6. Train
        7. Cross-Validation
        8. feature & label splitting
    """

    SHUFFLED = False

    def __init__(self, path, colnames=None, typ='csv'):
        """
        // TODO implement later on
        """
        if typ == 'csv':
            # If the data is in csv
            if not colnames:
                self._data = pd.read_csv(path)
            else:
                self._data = pd.read_csv(path, names=colnames, header=None)
        self._cv = None
        self._train = None

    @property
    def head(self):
        """
        Returns the internal dataframe head
        """
        return self._data.head(5)

    @property
    def tail(self):
        """
        Returns the internal dataframe tail
        """
        return self._data.tail(5)

    @property
    def size(self):
        """
        Returns the length of the dataframe
        """
        ret = "Rows/Entries: {}\nColumns/Variables: {}".format(self._data.shape[0], self._data.shape[1])
        return ret

    @property
    def train(self):
        """
        Return the train dataset
        """
        return self._train

    @property
    def test(self):
        """
        Return the cross-validation dataset
        """
        return self._cv

    def _shuffle(self, n_rounds=10, copy=False):
        """
        Shuffle the dataset and if
        copy = False stores back the result
        into the dataframe or else returns a
        new dataframe

        :param n_rounds: Number of times dataset is shuffled
        :param copy: Bool, if new dataset is created
        :return: pd.DataFrame or None
        """
        nrows = self._data.shape[0]
        index = np.arange(nrows)
        for _ in range(n_rounds):
            np.random.shuffle(index)
        if copy:
            Gather.SHUFFLED = True
            return self._data.iloc[index, :].reset_index(drop=True)
        else:
            Gather.SHUFFLED = True
            self._data = self._data.iloc[index, :].reset_index(drop=True)
            return
    
    def _split(self, split_ratio=0.4, dataset=None):
        """
        This splits the dataset into
        two parts for training and testing
        to avoid overfitting after ML fitting.
        """
        if not isinstance(dataset, pd.core.frame.DataFrame):
            dataSet = self._data
        else:
            dataSet = dataset
        if not Gather.SHUFFLED:
            warn('Splitting the dataset without shuffling!!', RuntimeWarning, stacklevel=2)
        split_index = ceil(dataSet.shape[0] * (1 - split_ratio))
        self._train = dataSet.iloc[0: split_index, :].reset_index(drop=True)
        self._cv = dataSet.iloc[split_index:, :].reset_index(drop=True)

    def train_test_split(self, label, n_rounds=10, copy=False, split_ratio=0.4, reject_cols=None, numpy_array=True):
        """
        Shuffle the data for ```n_rounds``` amount of times, split the
        dataset into train and test and then return feature and labels for
        both train and test.
        """
        all_features = set(self._data.columns.values)
        # The following feature can be better implemented using OrderedSet
        # however python doesn't support OrderedSet natively yet.
        # // TODO However https://stackoverflow.com/questions/1653970/does-python-have-an-ordered-set this may help
        
        # First create the feature and label cols
        if isinstance(reject_cols, set):
            final_reject_list = reject_cols.add(label)
        else:
            final_reject_list = {label}
        feature_cols = list(all_features - final_reject_list)
        label_col = label
        
        # Split and Shuffle the dataset
        if not copy:
            # Use the same dataset
            self._shuffle(n_rounds, copy)
            self._split(split_ratio)
        else:
            # Create a new dataset
            shuffled_data = self._shuffle(n_rounds, copy)
            self._split(split_ratio, shuffled_data)
        
        X_train = self.train.loc[:, feature_cols]
        X_test = self.test.loc[:, feature_cols]
        y_train = self.train.loc[:, label_col]
        y_test = self.test.loc[:, label_col]

        if numpy_array:
            return X_train.as_matrix(), X_test.as_matrix(), y_train.values, y_test.values
        else:
            return X_train, X_test, y_train, y_test

def main(path, cols=None):
    """
    The main calling method
    """
    gather = Gather(path, cols)
    a, b, c, d = gather.train_test_split(label='species')
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
