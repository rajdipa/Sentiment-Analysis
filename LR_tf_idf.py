# Implement the term frequency inverse document frequency (tf-idf) model for feature extraction, using nltk's countvectorizer package.
# Then implement Logistic Regression with built-in K-fold CV.
# Includes timer for cleaning and training time.

# https://scikit-learn.org/stable/modules/feature_extraction.html?fbclid=IwAR3A4frIuOR21nQqQGztVFw0UO1aEwgK3QqjpYNo3uFbmKmvPfi6oT0KgO4#tfidf-term-weighting

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import cleaning
import timeit
import numpy


# MAIN -----------------------------------------------

# Get data
phrases = cleaning.read_data("train.csv", "Phrase")
labels = cleaning.read_data("train.csv", "Sentiment")

# Tokenize and clean
start_time_clean = timeit.default_timer()
cleaned = cleaning.tokenize_data(phrases)
# cleaned = cleaning.stem_data(cleaned)
result = cleaning.filter_data(cleaned)
elapsed_time_clean = timeit.default_timer() - start_time_clean
print("Cleaning finished in " + str(elapsed_time_clean) + " seconds")


# Tf-idf model to extract features.
start_time_extract = timeit.default_timer()
transformer = TfidfTransformer(smooth_idf=False)

result = cleaning.untokenize(result)
result = numpy.array(result)
result = result.reshape(-1, 1) # convert to 2D array

X_train = transformer.fit(result).transform(result) # X_train.toarray()
elapsed_time_extract = timeit.default_timer() - start_time_extract
print("Feature extracting finished in " + str(elapsed_time_extract) + " seconds")


# Train logistic regression model with built-in K-fold CV.
start_time_train = timeit.default_timer()
# Try a variety of C-values
# param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
param_grid = {'C': [0.6, 0.8, 1, 2]}
# see hwk3-1 for optimal C value
# 5-fold CV because 10-fold takes too long
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, labels)
elapsed_time_train = timeit.default_timer() - start_time_train
print("Training finished in " + str(elapsed_time_train) + " seconds")

# Print Cross Validation estimates and optimal parameters.
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_) # Output best parameter
print("Best estimator: ", grid.best_estimator_)

# Creates a n by m word matrix of n phrases and m unique words.
# vect = CountVectorizer(max_features=1000)
# X = vect.fit_transform(data).toarray()
# print(X)

# Output: number of times each unique word appears in each phrase
# [[0 1 0 0]
#  [0 0 1 0]
#  [1 0 0 0]
#  [0 0 0 1]]
