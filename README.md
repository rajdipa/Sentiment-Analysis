Sentiment Analysis Project

Authors
----------------------------------------
Chowdhury Rajdipa

Li Sherri

Wang Jerry



Pre-requisites
----------------------------------------
1. Have Python 3+ installed.

2. Run
$ pip3 install nltk
$ pip3 install scikit-learn
$ pip3 install scipy

3. Go into python3 environment and run
```
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('words')
nltk.download('opinion_lexicon')
```
and other necessary packages when errors appear.



Training the Model and using the Model
----------------------------------------
Clone the repository to your machine.
```
$ cd SentimentAnalysis/scripts
$ python3 training_testing_LR_NB.py
$ python3 predict.py
```

You can view the cleaned data by running
```
$ python3 cleaning.py
```
You may substitute python with python3 if python is your path variable to running python3.
Some users with both python2 and python3 must type python3.



File Descriptions
----------------------------------------
cleaning.py includes functions to tokenize, clean, and untokenize the data files in the data folder [3].

training_testing_LR_NB.py imports module cleaning.py to pre-process both train and test data prior to feature extraction. It builds LR and NB models, and performs 5-fold CV [2] to get train accuracy.

predict.py trains a LR classifier with C=2 on the train data, and outputs the model predictions on testset_1.csv into the file 'output.csv'.

train.csv contains the corpus of labeled train data, which are phrases with their labeled sentiment, phraseId, and sentenceId [1].

testset_1.csv contains the corpus of unlabeled test data, the instances for which we want to predict sentiment (0 through 4).




References
----------------------------------------
1. Bo Pang and Lillian Lee. Seeing stars: Exploiting class relationships for sentiment categorization with respect to
rating scales. In ACL 2005, 43rd Annual Meeting of the Association for Computational Linguistics, Proceedings
of the Conference, 25-30 June 2005, University of Michigan, USA, pages 115–124, 2005.

2. Rajendran, Charles. Text classification using the Bag Of Words Approach with NLTK and Scikit Learn.
2018. Online. https://www.linkedin.com/pulse/text-classification-using-bag-words-approach-nltk-scikit-rajendran/

3. NLTK package. https://www.nltk.org/api/nltk.tokenize.html

4. SKlearn package. https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
