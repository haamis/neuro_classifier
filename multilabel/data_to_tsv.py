import pickle
from sklearn.model_selection import train_test_split
import os
import sys

def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)

abstracts = load_data(sys.argv[1])
labels = load_data(sys.argv[2])
abs_train, abs_test, lab_train, lab_test = train_test_split(abstracts, labels, test_size=0.2)
abs_test, abs_dev, lab_test, lab_dev = train_test_split(abs_test, lab_test, test_size=0.5)

with open(sys.argv[3] + "_train.tsv", "wt") as f:
    for abstract, label in zip(abs_train, lab_train):
        f.write(abstract + '\t' + ','.join(label) + '\t\n')

with open(sys.argv[3] + "_dev.tsv", "wt") as f:
    for abstract, label in zip(abs_dev, lab_dev):
        f.write(abstract + '\t' + ','.join(label) + '\t\n')

with open(sys.argv[3] + "_test.tsv", "wt") as f:
    for abstract, label in zip(abs_test, lab_test):
        f.write(abstract + '\t' + ','.join(label) + '\t\n')