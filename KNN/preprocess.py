import pandas as pd # For all data processing
import numpy as np

class Gather:
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
    """

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
        if self._train:
            return self._train
        else:
            raise AttributeError("Dataset is not split yet!!")

    @property
    def test(self):
        """
        Return the cross-validation dataset
        """
        if self._cv:
            return self._cv
        else:
            raise AttributeError("Dataset is not split yet!!")


    def shuffle(self, n_rounds=10, copy=False):
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
            return self._data.iloc[index, :].reset_index(drop=True)
        else:
            self._data = self._data.iloc[index, :].reset_index(drop=True)
            return
    
    def split(self, split_ratio=0.4):
        pass





def main(path, cols=None):
    """
    The main calling method
    """
    gather = Gather(path, cols)
    # print(gather.head)
    # print(gather.tail)
    # print(gather.size)
    print(gather.shuffle(copy=True))
    gather.split()
    print(gather.train)
    print(gather.test)

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
