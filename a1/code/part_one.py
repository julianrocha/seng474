from sklearn import tree
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


data = np.genfromtxt('data/cleaned_processed.cleveland.data',delimiter=',')

X, y = data[:,:-1], data[:,-1] # split data into inputs X and labels y
# Train/test split example : https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html?highlight=train%20test%20split#sklearn.model_selection.train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# SCALE???
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# https://scikit-learn.org/stable/modules/tree.html
dt_classifier = tree.DecisionTreeClassifier(criterion='gini',random_state=42) # could also use 'entropy'
dt_classifier = dt_classifier.fit(X_train,y_train)
# try different split criterion 
# try different pruning values/techniques
print("Train acc no pruning:",dt_classifier.score(X_train, y_train))
print("Test acc no pruning:",dt_classifier.score(X_test, y_test))

# https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html#sphx-glr-auto-examples-tree-plot-cost-complexity-pruning-py
# Post pruning decision trees with cost complexity pruning
path = dt_classifier.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

clfs = []
for ccp_alpha in ccp_alphas:
    clf = tree.DecisionTreeClassifier(random_state=42, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)

clfs = clfs[:-1]            # remove last classifier which only has one node
ccp_alphas = ccp_alphas[:-1]# remove last classifier which only has one node

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

# Plot for choice of cpp_alpha
fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

dt_classifier = tree.DecisionTreeClassifier(random_state=42, ccp_alpha=0.025)
dt_classifier = dt_classifier.fit(X_train,y_train)
print("Train acc:",dt_classifier.score(X_train, y_train))
print("Test acc:",dt_classifier.score(X_test, y_test))

# Random Forest
clfs = []
n_estimators = range(1,100)
for i in n_estimators:
    clf = RandomForestClassifier(random_state=42,n_estimators=i)
    clf.fit(X_train, y_train)
    clfs.append(clf)

train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]
# Plot for choice of n_estimators
fig, ax = plt.subplots()
ax.set_xlabel("n_estimators")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs n_estimators for training and testing sets")
ax.plot(n_estimators, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(n_estimators, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

clf = RandomForestClassifier(random_state=42,n_estimators=28)
clf.fit(X_train, y_train)
print("Train acc:",clf.score(X_train, y_train))
print("Test acc:",clf.score(X_test, y_test))