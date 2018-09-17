import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, SelectPercentile ,f_classif
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from enum import Enum
import numpy as np

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

class fs_types(Enum):
    KBest = 1
    Percentile = 2

fs_switch = {
    1: SelectKBest(f_classif, k=2),
    2: SelectPercentile(f_classif, percentile=5)
}

fs_param_switch = {
    1: lambda X: list(map(lambda x: {'fs__k': x}, list(range(1, len(X[0]), len(X[0])//15)))),
    2: lambda X: list(map(lambda x: {"fs__percentile": x},list((1, 3, 6, 10, 15, 20, 30, 40, 60, 80, 100))))
}

class clf_types(Enum):
    KNneighbour = 1
    clf = 2

clf_switch = {
    1: KNeighborsClassifier(n_neighbors=1),
    2: SVC(C=1.0)
}

clf_param_switch = {
    1: list(range(1,11,1)),
    2: np.linspace(0.1,1,10)
}

def findBestParameters(X, y, feature_selector_type, classifier_type):

    feature_selector = fs_switch.get(feature_selector_type)
    classifier = clf_switch.get(classifier_type)
    fs_params = fs_param_switch.get(feature_selector_type)(X)

    if any(map(lambda x: x is None, [feature_selector, classifier])):
        print("Error, invalid feature_selector or classifier type\n")
        exit()

    pipeline = Pipeline([("fs", feature_selector), ("clf", classifier)])

    scores_means = []
    scores_stderrs = []
    selected_features = []

    for param in fs_params:
        print("Training on {}".format(param))
        # print(pipeline.get_params())
        pipeline.set_params(**param)
        # print(pipeline.get_params())
        curr_scores = cross_val_score(pipeline, X_train, y_train, n_jobs=1, cv=5)
        # print(curr_scores)
        scores_means.append(curr_scores.mean())
        scores_stderrs.append(curr_scores.std())
        pipeline.named_steps.fs.fit(X_train, y_train)
        selected_features.append(pipeline.named_steps.fs.get_support())

    print("Mean scores: {}".format(scores_means))
    print("Std Errors of mean scores: {}".format(scores_stderrs))


# Main run
data = load_breast_cancer(return_X_y=False)
printBasicInfo(data)

X = data.data #, new_feature_names = dropErrorColumns(data.data, data.feature_names)
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

findBestParameters(X_train, y_train, 1, 1)

# for features in selected_features:
#     print("{}\n".format(features))