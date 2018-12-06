# Make predictions on testing data using our LR model.

import training
import cleaning
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
# Call function that builds model on train.csv and uses it to test data.

model = training.main()
vect = CountVectorizer(min_df=2, ngram_range=(0, 30))

phrases = cleaning.read_data("testset_1.csv", "Phrase")
ids = cleaning.read_data("testset_1.csv", "PhraseId")
tokenized = cleaning.tokenize_data(phrases)
filtered = cleaning.filter_data(tokenized)
prepared = training.untokenize(filtered)
testdata = vect.fit_transform(prepared)

predictions = model.predict(testdata)

data = create_df(ids, predictions)
df = pd.DataFrame({'PhraseId':np.array(ids),'Sentiment':np.array(predictions)})
outfile = "ChowdhuryLiWang.csv"
df.to_csv(outfile, index=False)
