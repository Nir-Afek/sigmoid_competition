import pandas as pd
from sklearn.tree import DecisionTreeClassifier

train = pd.read_csv("test-input.csv")
X = train.drop('slab_released', axis=1)
y = train['slab_released']

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X, y)

# Just check the features and their first letter to get the word
features = clf.tree_.feature[clf.tree_.feature >= 0]
result = ''.join([name[0] for name in X.columns[features]]).upper()

with open("test-output.txt", "w") as file:
    file.write(f"SIGMOID_{result}")