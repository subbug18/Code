from sklearn.datasets import load_iris
from matplotlib import  pyplot as plt
import  numpy as np

data = load_iris()
features = data['data']
features_names = data['feature_names']
target = data['target']

for t,marker,c in zip(xrange(3),">ox","rgb"):
    # We plot each class on its own to get different colored markers
    plt.scatter(features[target == t,0],
    features[target == t,1],
    marker=marker,
            c=c)

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df=1)
print(vectorizer)

content = ["How to format my hard disk", " Hard disk format problems "]
X = vectorizer.fit_transform(content)
print(vectorizer.get_feature_names())

print(X)
print(X.toarray().transpose())
