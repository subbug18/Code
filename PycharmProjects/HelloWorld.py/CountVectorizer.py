from sklearn.feature_extraction.text  import CountVectorizer


import os

import nltk.stem

DIR = "/home/subbu/PycharmProjects/Image_Files"

posts = [open(os.path.join(DIR, f)).read() for f in os.listdir(DIR)]

#vectorizer = CountVectorizer(min_df=1)


english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCounterVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCounterVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedCounterVectorizer(min_df=1, stop_words='english')


X_train = vectorizer.fit_transform(posts)

print("X_train", X_train)
num_samples, num_features = X_train.shape

print("samples#:",num_samples,"features#", num_features)

print (vectorizer.get_feature_names())

print("posts")
print(posts)


new_post = "imaging databases interesting"
new_post_vc = vectorizer.transform([new_post])
print('new post')
print(new_post_vc)
print(new_post_vc.toarray())



import scipy as sp
def dist(v1, v2):
    delta = v1-v2
    return sp.linalg.norm(delta.toarray())

import sys
best_doc = None
best_dist= sys.maxint
best_i = None
for i in range(0, num_samples):
    post = posts[i]
    if posts==new_post:
        continue
    post_vec = X_train.getrow(i)
    d = dist(post_vec, new_post_vc)
    print( "=== Post %i with dist=%.2f: %s" % (i, d, posts))
    if d<best_dist:
        best_dist=d
        best_i=i

print("Best post is %i with dist=%.2f"%(best_i, best_dist))


s = nltk.stem.SnowballStemmer('english')
print(s.stem('image'))
print(s.stem('imaging'))
print(s.stem('imagination'))
print(s.stem('imagine'))


print(s.stem('buying'))
print(s.stem('boughts'))
print(s.stem('buys'))




