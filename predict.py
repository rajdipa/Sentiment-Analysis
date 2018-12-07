# Make predictions on testing data using our LR model.

import training
import cleaning
import pandas as pd
import numpy as np
import pickle as rick
import os.path
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# Check whether model has been built already.
if not (os.path.exists('dumped_model_LR.pkl')):
    # Call function that builds model on train.csv and uses it to test data.
    training.main()

# Open saved model.
model = 0
with open('dumped_model_LR.pkl', 'rb') as f:
    model = rick.load(f)

# Read in test data.
test_phrases = cleaning.read_data("testset_1.csv", "Phrase")
test_ids = cleaning.read_data("testset_1.csv", "PhraseId")

# Clean test data.
test_phrases = cleaning.tokenize_data(test_phrases)
test_phrases = cleaning.filter_data(test_phrases)
test_phrases = training.untokenize(test_phrases)
test_phrases = np.array(test_phrases)
#test_phrases = test_phrases.reshape(-1,1)
#test_phrases = test_phrases.reshape(109242,1)
# Counts phrases between 0 through 30 words.
#vect = CountVectorizer(min_df=2, ngram_range=(0, 30))
#test_phrases = vect.transform(test_phrases)
print(test_phrases.shape)
#print(test_phrases.shape[1])
test_phrases = test_phrases.reshape(1,-1)
# Predict labels on test data.
predictions = model.predict(test_phrases) # ValueError: X has 1 features per sample; expecting 170735

#data = create_df(ids, predictions)
df = pd.DataFrame({'PhraseId':np.array(ids),'Sentiment':np.array(predictions)})
outfile = "output.csv"
df.to_csv(outfile, index=False)
