##Purpose:
The Python code "RankNet_version_1.py" demonstrates how to implement RankNet algorithm which serves as a search engine optimizer or a ranking system of relevant products/URLS(search results).

##Data:
The data of search results is stored in the form of matrices, where one matrix corresponds to a query,
with the last column of the matrix representing the relevance scores of URLS
and the other columns being the features of URLS, which determines the relevant scores.
For simplicity, we randomly generate the feature matrix and
assume the ranking score is the absolute value of the sine of the sum of the square of the feature values.

##Code:
The main code of this project is RankVet_Version1.py

##Task:
The task is to establish a neural network model RankNet that explains the subtle
relationship between the features of a URL and its relevant score.
Then we can use the RankNet to rank the URLS in terms of their relevant scores
so the most relevant URLS will appear on top of less relevant URLS.

##Method:
The method follows the RankNet algorithm described in Section 2: RankNet in Burges' paper
quoted below.
The RankNet object used in this code does not rely on
any pre-defined deep learning package but rather is built from scratch
though some of the classes/functions
such as the sigmoid activation class are borrowed from the 2nd reference quoted below.

##Results:
1."Decreasing_Cross_Entropy.png" shows the model gets better in ranking in terms of cross entropy loss with the training data.
2."Test_10_URLS" shows the ranking of test URLS (if the test data is consisted of 10 URLs) by RankNet in comparison with its actual ranking.
3."Test_5_URLS" shows the ranking of test URLS (if the test data is consisted of 5 URLs) by RankNet in comparison with its actual ranking.

##Reference:
1. From RankNet to LambdaRank to LambdaMART: An Overview
Christopher J.C. Burges
Microsoft Research Technical Report MSR-TR-2010-82
2. Neural Networks from Scratch in Python, Harrison Kingsley and Daniel Kukiela


