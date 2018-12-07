from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import cleaning

def untokenize(texts):
    docs = []
    for doc in texts:
        temp = ""
        for word in doc:
            temp += word + " "
        docs.append(temp)
    return docs

phrases = cleaning.read_data("train.csv", "Phrase")
labels = cleaning.read_data("train.csv", "Sentiment")
cleaned = cleaning.tokenize_data(phrases)
result = cleaning.filter_data(cleaned)



vect = CountVectorizer(min_df=2, ngram_range=(0, 30))
X_train = vect.fit(untokenize(result)).transform(untokenize(result))

param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid.fit(X_train, labels)
print("Best cross-validation score for LogisticRegression: {:.2f}".format(grid.best_score_))



param_grid = {'alpha': [0.001, 0.01, 0.1, 1,10],"fit_prior":[True, False]}
gs_clf = GridSearchCV(MultinomialNB(),param_grid, cv=5)
gs_clf.fit(X_train, labels)
print("Best cross-validation score for MultinomialNB: {:.2f}".format(gs_clf.best_score_))
