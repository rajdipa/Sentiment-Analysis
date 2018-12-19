# training_testing_LR_NB.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import cleaning

# Read in train data
phrases = cleaning.read_data("../data/train.csv", "Phrase")
labels = cleaning.read_data("../data/train.csv", "Sentiment")
cleaned = cleaning.tokenize_data(phrases)
result = cleaning.filter_data(cleaned)
result = cleaning.untokenize(result)

# Initialize Bag of Words feature extraction model
vect = CountVectorizer(min_df=2, ngram_range=(0, 30))
X_train = vect.fit(result).transform(result)

# Train Logistic Regression model
#param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
param_grid = {'C': [1]}# 1 turned out best
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, labels)
print("Best cross-validation score for LogisticRegression: {:.2f}".format(grid.best_score_))

# Train Naive Bayes model
# param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100],"fit_prior":[True, False]}
param_grid = {'alpha': [1]}# 1 turned out best
gs_clf = GridSearchCV(MultinomialNB(),param_grid, cv=5)
gs_clf.fit(X_train, labels)
print("Best cross-validation score for MultinomialNB: {:.2f}".format(gs_clf.best_score_))
