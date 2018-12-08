# Implement the bag of words model for feature extraction, using nltk's countvectorizer package.
# Then implement Logistic Regression with built-in K-fold CV.
# Includes timer for cleaning and training time.

# Call main() at bottom

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import cleaning
import timeit
import pickle as rick
import os.path
import pandas as pd
import numpy as np


# MAIN -----------------------------------------------
def main():
    # Get train data.
    train_phrases = cleaning.read_data("train.csv", "Phrase")
    train_labels = cleaning.read_data("train.csv", "Sentiment")

    # Tokenize and clean train data prior to building model.
    start_time_clean = timeit.default_timer()
    train_phrases = cleaning.tokenize_data(train_phrases)
    # Stemming wasn't helpful.
    # train_phrases = cleaning.stem_data(train_phrases)
    train_phrases = cleaning.filter_data(train_phrases)
    elapsed_time_clean = timeit.default_timer() - start_time_clean
    print("Cleaning finished in " + str(elapsed_time_clean) + " seconds")

    # Bag of Words model to extract features.
    start_time_extract = timeit.default_timer()
    # Only uses the phrases between 0 through 30 words.
    vect = CountVectorizer(min_df=2, ngram_range=(1, 30))
    train_phrases = cleaning.untokenize(train_phrases)
    train_phrases = vect.fit(train_phrases).transform(train_phrases)
    elapsed_time_extract = timeit.default_timer() - start_time_extract
    print("Feature extracting finished in " + str(elapsed_time_extract) + " seconds")

    # Train logistic regression model with built-in K-fold CV.
    start_time_train = timeit.default_timer()
    # Try a variety of C-values
    #param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]} # best is 1
    #param_grid = {'C': [0.6, 0.8, 1, 2]} # best is 2
    #param_grid = {'C': [5, 7, 9, 11, 13]} # best is 5 for cv=10
    param_grid = {'C': [2]}
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(train_phrases, train_labels)
    elapsed_time_train = timeit.default_timer() - start_time_train
    print("Training finished in " + str(elapsed_time_train) + " seconds")

    # Print Cross Validation estimates and optimal parameters.
    # print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    # print("Best parameters: ", grid.best_params_) # Output best parameter
    # print("Best estimator: ", grid.best_estimator_)
    model = grid.best_estimator_

    # Read in test data.
    test_phrases = cleaning.read_data("testset_1.csv", "Phrase")
    test_ids = cleaning.read_data("testset_1.csv", "PhraseId")

    # Clean test data.
    test_phrases = cleaning.tokenize_data(test_phrases)
    test_phrases = cleaning.filter_data(test_phrases)
    test_phrases = cleaning.untokenize(test_phrases)
    test_phrases = vect.transform(test_phrases)

    print(test_phrases.shape)

    # Predict labels on test data.
    predictions = model.predict(test_phrases)

    # Output predictions.
    df = pd.DataFrame({'PhraseId':np.array(test_ids),'Sentiment':np.array(predictions)})
    outfile = "output.csv"
    df.to_csv(outfile, index=False)


main()
