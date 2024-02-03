'''
Input: a list of (relevance scores)
Output: List the index and content of the elements from high score to low score
        It displays as a pandas dataframe.
Example: [5,2,3,4,1] resulting in [(0,5),(3,4),(2,3),(1,2),(4,1)],
         the first element in a tuple is the original position
         of the content in the input list;
         the second element in a tuple is the actual content


'''

import numpy as np
import pandas as pd

def rank_scores(relevance_scores):
    relevance_scores_with_index = [0,0]
    for index, item in enumerate(relevance_scores):
        relevance_scores_with_index = np.vstack((relevance_scores_with_index,[index,item]))

    relevance_scores_with_index = pd.DataFrame(relevance_scores_with_index)
    relevance_scores_with_index = relevance_scores_with_index.iloc[1:,:]
    relevance_scores_with_index.columns = ['Id','Scores']
    relevance_scores_with_index = relevance_scores_with_index.sort_values(by='Scores', ascending = False)
    return relevance_scores_with_index

