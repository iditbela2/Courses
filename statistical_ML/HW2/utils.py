import requests
import pandas as pd

from sklearn.datasets import fetch_mldata


def load_mnist():
    try:
        mnist = fetch_mldata('MNIST original')

        data_df = pd.DataFrame(mnist['data'])
        label_df = pd.DataFrame(mnist['target'])

        return data_df, label_df
    except requests.exceptions.RequestException:
        print('HTTP exception, check you connection and try again')
