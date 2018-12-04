import pandas as pd

# read in the csv into a dataframe object
data = pd.read_csv("train.csv")
# print first 5 rows of data
print(data.head())
# print the sentiment column
print(data.loc[:, "Sentiment"])
# another way to print the sentiment column
print(data[["Sentiment"]])
