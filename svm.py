# Practice using SVM to classify data.

from sklearn import svm

X = [[0,0],[1,1]]
y = [0,1]
clf = svm.SVC(gamma='scale')
clf.fit(X,y)
clf.support_vectors_
