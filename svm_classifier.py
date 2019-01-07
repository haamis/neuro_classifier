import json, sys, re

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, recall_score, make_scorer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

def preprocess_data(input_file):
    print("Reading input file..")

    with open(input_file) as f:
        data = json.load(f)

    abstracts = []
    is_neuro = []

    # Neuroscience MeSH-terms.
    reg = re.compile(r"(D009420|D009457|D009474|D009422|D001520|D011579|D001523|D004191)")

    for article in tqdm(data, desc="Grabbing abstracts and mesh terms"):
        abstracts.append("\n".join([x["text"] for x in article["abstract"]]))
        is_neuro.append(0)
        for mesh_term in article["mesh_list"]:
            if reg.match(mesh_term["mesh_id"]):
                is_neuro[-1] = 1
                break

    return (abstracts, is_neuro)

(x, y) = preprocess_data("./" + sys.argv[1])

print("Running vectorizer..")
vectorizer = CountVectorizer()
x = vectorizer.fit_transform(x)

print("Running tfidf..")
tfidf_transformer = TfidfTransformer()
x = tfidf_transformer.fit_transform(x)

rus = RandomUnderSampler(random_state=0)
x, y = rus.fit_resample(x,y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

svc = LinearSVC(max_iter=1500)
#print(svc.get_params().keys())

classifier = GridSearchCV(svc, {'C': [1, 10, 100, 1000, 10000, 100000, 1000000],},
                            scoring=make_scorer(f1_score), n_jobs=20)

classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))