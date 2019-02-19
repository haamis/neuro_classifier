import sys, pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score

def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)

print("Reading input files..")
x = load_data("./" + sys.argv[1])
y = load_data("./" + sys.argv[2])

print("Running vectorizer..")
vectorizer = TfidfVectorizer(min_df=3)
x = vectorizer.fit_transform(x)

print("Binarizing labels..")
mlb = MultiLabelBinarizer(sparse_output=True)
y = mlb.fit_transform(y)

print("Splitting..")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
del x,y # Free up ~3 gigabytes (with full dataset).

classifier = GridSearchCV(OneVsRestClassifier(LinearSVC(), n_jobs=20), {'estimator__C': [0.0001, 0.001, 0.01, 0.1]}, cv=5, scoring='f1_micro')

print("Classifying..")

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

print(y_pred[:])

print("Precision:", precision_score(y_test, y_pred, average='micro'))
print("Recall:", recall_score(y_test, y_pred, average='micro'))
print("F1:", f1_score(y_test, y_pred, average='micro'))
