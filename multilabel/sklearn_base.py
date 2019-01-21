import sys, pickle

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.preprocessing import OrdinalEncoder
import numpy


def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)

print("Reading input files..")
x = load_data("./" + sys.argv[1])
y = load_data("./" + sys.argv[2])

#print("Splitting..")
#x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
#del x,y # Free up ~3 gigabytes (with full dataset).

print("Running..")

classifier = MLPClassifier()
ord_enc = OrdinalEncoder()

x = numpy.array(x).reshape(-1, 1)
#y = numpy.array([y])
x = ord_enc.fit_transform(x, y)

classifier.fit(x, y)

#y_pred = classifier.predict(x_test)
#print("Precision:", precision_score(y_test, y_pred))
#print("Recall:", recall_score(y_test, y_pred))
#print("F1:", f1_score(y_test, y_pred))