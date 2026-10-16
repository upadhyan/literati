"""Fit Literati on a dataset and plot the tree. Edit the settings below."""
from sklearn.metrics import accuracy_score

from data_utils import DataFactory_clf
from literati import LiteratiClassifier
from plot_tree import plot_literati_tree

DATASET = "rice"
DEPTH = 3
MAX_ORDER = 3
TIME_LIMIT = 60
LAMBDA_ = 0.0 # penalty on the number of internal nodes in the tree. Set to 0 for no penalty.
ALPHA_ = 0.0 # penalty on the shape function complexity used in the tree. Set to 0 for no penalty.

df = DataFactory_clf(DATASET)
X_train, y_train, X_val, y_val, X_test, y_test = df.get_data(fold_idx=0)

clf = LiteratiClassifier(max_depth=DEPTH, 
                         max_order=MAX_ORDER,
                         lambda_=LAMBDA_,
                         alpha_ = ALPHA_,
                         time_limit=TIME_LIMIT)
clf.fit(X_train, y_train, feature_dict=df.feature_dict)

print("train acc:", accuracy_score(y_train, clf.predict(X_train)))
print("test acc: ", accuracy_score(y_test, clf.predict(X_test)))
print("optimal:  ", clf.is_solved())

fig = plot_literati_tree(clf, class_names=[str(c) for c in df.class_names], title=DATASET)
fig.savefig("tree.png", dpi=150, bbox_inches="tight")
print("saved tree.png")
