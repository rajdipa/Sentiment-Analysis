# Implement the bag of words model for feature extraction, using nltk's countvectorizer package.
# Then implement Logistic Regression with built-in K-fold CV.
# Includes timer for cleaning and training time.

# Call training.main() from the calling file, which will be predict.py

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import cleaning
import timeit
import pickle as rick

# You must un-tokenize the words back into phrases before running the model.
def untokenize(texts):
    docs = []
    for doc in texts:
        temp = ""
        for word in doc:
            temp += word + " "
        docs.append(temp)
    return docs

# MAIN -----------------------------------------------
def main():
    # Get data
    phrases = cleaning.read_data("train.csv", "Phrase")
    labels = cleaning.read_data("train.csv", "Sentiment")

    # Tokenize and clean
    start_time_clean = timeit.default_timer()
    cleaned = cleaning.tokenize_data(phrases)
    # cleaned = stem_data(cleaned)
    result = cleaning.filter_data(cleaned)
    elapsed_time_clean = timeit.default_timer() - start_time_clean
    print("Cleaning finished in " + str(elapsed_time_clean) + " seconds")


    # Bag of Words model to extract features.
    start_time_extract = timeit.default_timer()
    vect = CountVectorizer(min_df=2, ngram_range=(1, 30))
    result = untokenize(result)
    X_train = vect.fit(result).transform(result)
    elapsed_time_extract = timeit.default_timer() - start_time_extract
    print("Feature extracting finished in " + str(elapsed_time_extract) + " seconds")


    # Train logistic regression model with built-in K-fold CV.
    start_time_train = timeit.default_timer()
    # Try a variety of C-values
    #param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]} # best is 1
    #param_grid = {'C': [0.6, 0.8, 1, 2]} # best is 2
    #param_grid = {'C': [5, 7, 9, 11, 13]} # best is 5 for cv=10
    param_grid = {'C': [2]}
    # Use 5-fold CV because 10-fold takes too long.
    grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid.fit(X_train, labels)
    elapsed_time_train = timeit.default_timer() - start_time_train
    print("Training finished in " + str(elapsed_time_train) + " seconds")

    # Print Cross Validation estimates and optimal parameters.
    # print("Best cross-validation score: {:.2f}".format(grid.best_score_))
    # print("Best parameters: ", grid.best_params_) # Output best parameter
    # print("Best estimator: ", grid.best_estimator_)

    # Retrun the LR model
    # return grid.best_estimator_

    # Save model into pickle file to be used later.
    with open('dumped_model_LR.pkl', 'wb') as f:
        rick.dump(grid.best_estimator_, f)
