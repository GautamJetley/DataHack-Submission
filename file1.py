import pandas as pd

train_features = pd.read_csv('/mnt/data/training_set_features.csv')
train_labels = pd.read_csv('/mnt/data/training_set_labels.csv')
test_features = pd.read_csv('/mnt/data/test_set_features.csv')


train_features.head(), train_labels.head(), test_features.head()
