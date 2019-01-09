import sys, pickle

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC, SVC
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.feature_extraction.text import TfidfVectorizer

from imblearn.under_sampling import RandomUnderSampler

def load_data(file_name):
    
    with open(file_name, "rb") as f:
        return pickle.load(f)

print("Reading input files..")
x = load_data("./" + sys.argv[1])
y = load_data("./" + sys.argv[2])

print("Running vectorizer..")
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(x)

print("Splitting..")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print("Undersampling..")
rus = RandomUnderSampler(random_state=0)
x_train, y_train = rus.fit_resample(x_train,y_train)

print("Running..")

# Still gives ConvergenceWarnings with complete_output.
#svc = LinearSVC(max_iter=100000)#, class_weight='balanced')
svc = SVC(kernel='linear', max_iter=100000)

# TODO: Try with lower C too.
classifier = GridSearchCV(svc, {'C': [1, 10, 100, 1000, 10000]}, cv=5,
                            scoring='f1', n_jobs=20)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))