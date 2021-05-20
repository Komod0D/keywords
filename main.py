from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

from nltk.corpus import stopwords

import pandas as pd

labels = ["API", "Blockchain", "Compliance", "Data/ML", "Development", "HR", "Infrastructure",
          "Monetization", "Productivity", "UI", "Security"]

labelled_df = pd.read_csv("labelled.csv")


labelled_df.dropna(axis=0, inplace=True, subset=["repo", "text"])
labelled_df.reset_index(inplace=True, drop=True)


unlabelled_df = pd.read_csv("unlabelled.csv").head(100)
# unlabelled_df.tail(-100).to_csv("unlabelled.csv")
unlabelled_df.dropna(axis=0, inplace=True, subset=["repo", "text"])
unlabelled_df.reset_index(drop=True, inplace=True)

categories = [" ".join(labelled_df[labelled_df["label"] == label]["text"].tolist()) for label in labels]

print("Fitting CV & TFIDF")

corpus = labelled_df["text"]
eng = stopwords.words("english")

cv = CountVectorizer(stop_words=eng, min_df=0.05, max_df=0.5)
tfidf = TfidfTransformer()

arr = cv.fit_transform(corpus)
arr = tfidf.fit_transform(arr)
arr = np.array(arr.todense())

unlabelled_arr = tfidf.transform(cv.transform(unlabelled_df["text"].tolist()))
unlabelled_arr = np.array(unlabelled_arr.todense())

n_components = 75
svd = TruncatedSVD(n_components=n_components, n_iter=20)

svd.fit(np.vstack((arr, unlabelled_arr)))

explained_variance = np.sum(svd.explained_variance_ratio_)
print(f"Explained variance with {n_components} components: {explained_variance * 100}%")

X = arr
Y = np.zeros(shape=(arr.shape[0],))

for i, row in labelled_df.iterrows():
    label = row["label"]
    if not pd.isna(label):
        Y[i] = labels.index(label)

X_out = unlabelled_arr


min_test, min_train = 20, 50
accuracies = []
for n_train in range(min_train, len(X) - min_test):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=n_train)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, Y_train)

    Y_pred = knn.predict(X_test)

    # Y_pred = np.hstack([p[:, 1:] for p in Y_pred])
    diff = np.where(Y_pred != Y_test, 1, 0)
    mistakes = diff.sum()
    total = Y_test.size
    accuracy = 1. - mistakes / total
    accuracies.append(accuracy)

plt.plot(np.arange(min_train, len(X) - min_test), accuracies, "r-")
plt.show()


"""
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, Y)

Y_pred = knn.predict(X_out)

unlabelled_df["label"] = pd.Series(map(lambda pred: labels[int(pred)], Y_pred))

unlabelled_df = unlabelled_df[["repo", "label"]]
unlabelled_df.to_csv("repopulated_sample.csv", index=False)
"""