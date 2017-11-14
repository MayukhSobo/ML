import sys

from knn import KNN


def main(path):
    data_params = {
        'label': 'species'
    }
    cols = [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'species'
    ]
    knn = KNN(k=3, path=path, colnames=cols, **data_params)
    knn.apply()


if __name__ == '__main__':
    pth = sys.argv[1]
    main(pth)
