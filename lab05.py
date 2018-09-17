import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

def printDemo(example, data):
    print("The first example of the data:\n")
    for i, feature_name in enumerate(data.feature_names):
        print("{}: {}".format(feature_name, example[i]))
    print("\n")

def printBasicInfo(data):
    print("Shape of dataset: {}\n", data.data.shape)
    print("Length of examples: {}\n".format(len(data.data))) #Length of data
    print("Attributes: {}\n".format(data.feature_names)) #Attributes
    print("Classes: {}\n".format(data.target_names)) #Classes
    printDemo(data.data[0], data)

def dropErrorColumns(data, feature_names):
    df = pd.DataFrame(data=data)
    errorColumns = [index for index, column in enumerate(feature_names) if "error" in column]
    # print(errorColumns)
    df.drop(labels=errorColumns, axis=0)
    new_feature_names = [n for i, n in enumerate(feature_names) if i not in errorColumns]
    return df.values, new_feature_names

# Main run
data = load_breast_cancer(return_X_y=False)
printBasicInfo(data)

X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

selection_ks = [2]
selection_ks.extend(list(range(5, 31, 5)))

feature_selector = SelectKBest(chi2, k=2)

chi_svc = Pipeline([("chi", feature_selector), ("svc", SVC(C=1.0))])

scores_means = []
scores_stderrs = []

for k in selection_ks:
    print("Training on k={}".format(k))
    chi_svc.set_params(chi__k=k)
    curr_scores = cross_val_score(chi_svc, X_train, y_train, n_jobs=1)
    print(curr_scores)
    scores_means.append(curr_scores.mean())
    scores_stderrs.append(curr_scores.std())

print("Mean scores: {}".format(scores_means))
print("Std Errors of mean scores: {}".format(scores_stderrs))