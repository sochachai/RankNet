'''
Purpose:
This Python code serves as a search engine optimizer or a ranking system of relevant products/URLS(search results).

Data:
The data of search results is stored in the form of matrices, where one matrix corresponds to a query,
with the last column of the matrix representing the relevance scores of URLS
and the other columns being the features of URLS, which determines the relevant scores.
For simplicity, we randomly generate the feature matrix and
assume the ranking score is the absolute value of the sine of the sum of the square of the feature values.


Task:
The task is to establish a neural network model RankNet that explains the subtle
relationship between the features of a URL and its relevant score.
Then we can use the RankNet to rank the URLS in terms of their relevant scores
so the most relevant URLS will appear on top of less relevant URLS.

Method:
The method follows the RankNet algorithm described in Section 2: RankNet in Burges' paper
quoted below.
The RankNet object used in this code does not rely on
any pre-defined deep learning package but rather is built from scratch
though some of the classes/functions
such as the sigmoid activation class are borrowed from the 2nd reference quoted below.


Reference:
1. From RankNet to LambdaRank to LambdaMART: An Overview
Christopher J.C. Burges
Microsoft Research Technical Report MSR-TR-2010-82

2. Neural Networks from Scratch in Python, Harrison Kingsley and Daniel Kukiela
'''
import pandas as pd
import numpy as np
import Get_ranking_from_scores
np.random.seed(0)



'''
Construct neural networks from scratch
'''
class Matrix_Layer:
    def __init__(self, row_number, col_number, sigma, learning_rate):
        '''
        :param row_number: the row dimension of the matrix
        :param col_number: the column dimension of the matrix
        :param sigma: scaling constant of the cross entropy loss
        :param learning_rate: learning rate of gradient decent
        '''
        self.weights = np.random.random((row_number, col_number))
        self.bias = np.random.random((1, col_number))
        self.sigma = sigma
        self.learning_rate = learning_rate

    def forward_prediction(self, inputs):
        self.inputs = inputs # need the inputs for back_propagation
        self.outputs = np.dot(self.inputs, self.weights)
        self.outputs = self.outputs + self.bias
        return None

    def backward_propagation(self, dvalues):
        '''
        :param inputs: outputs of the previous layer
        :param dvalues: derivatives of the next layer, each column corresponds to a neuron of the next layer
        '''
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbias = np.sum(dvalues, axis = 0, keepdims = True) # double brackets, row vector formatted as matrix
        self.dinputs = np.dot(dvalues, self.weights.T) # need this to pass to previous layer
        self.weights = self.weights - self.learning_rate * self.dweights
        self.bias = self.bias - self.learning_rate * self.dbias
        return None

class Relu_Activation:
    def __init__(self, slope, bias):
        self.slope = slope
        self.bias = bias

    def forward_prediction(self, inputs):
        self.inputs = inputs # need the inputs for back_propagation
        self.outputs = self.slope * (np.maximum(inputs, self.bias) - self.bias) # note it is np.maximum not np.max!
        return None

    def backward_propagation(self, dvalues):
        self.dinputs = dvalues.copy() * self.slope
        self.dinputs[self.inputs <= self.bias] = 0
        return None


class Sigmoid_Activation:
    def __init__(self, mu):
        self.mu = mu

    def forward_prediction(self, inputs):
        self.inputs = inputs # need the inputs for back_propagation
        self.outputs = (lambda x: 1 / (1 + np.exp(-x/self.mu)))(inputs)
        return None

    def backward_propagation(self, dvalues):
        self.dinputs = (lambda x: (np.exp(x/self.mu)/self.mu)/ (1 + np.exp(x/self.mu))**2)(dvalues)
        return None


class Softmax_Activation:
    def forward_prediction(self, inputs):
        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities

    def backward_propagation(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)
        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1) # one sample of softmax output, i.e. S_{i}
                                                         # reshape as a column vector
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            # S_{i,j}delta_{j,k} - S_{i,j}S_{i,k}
            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
            # 1. Each sample corresponds to a fixed i
            # 2. The target is to get for each k (the index of input features):
            # \sum_{j}(\partial{f}/\partial{S_{i,j}}) \times (\partial{S_{i,j}}/\partial{z_{i,k}})

            # 3. Each row of Jacobian matrix corresponds to a fixed j and
            # a vector of \partial{S_{i,j}}/\partial{z_{i,k}} if all k's are listed
            # Note the Jacobian matrix is symmetric

            # 4. single_dvalues corresponds to \partial{f}/\partial{S_{i,j}}
            # 5. f is the output of the Softmax layer or the input of the next layer



class Neural_Network:
    def __init__(self, X_train, y_train, X_test, y_test, sigma, learning_rate):
        '''
        :param X_train: training set features
        :param y_train: training set labels
        :param X_test: testing set features
        :param y_test: testing set labels
        :param sigma: parameter of loss function
        :param learning_rate: scaling factor of the gradient (common in most ML models)
        '''
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.Net_1 = Matrix_Layer(self.X_train.shape[1], 10, sigma, learning_rate)
        self.Softmax_1 = Softmax_Activation()
        self.Net_2 = Matrix_Layer(10, 5, sigma, learning_rate)
        self.Softmax_2 = Softmax_Activation()
        self.Net_3 = Matrix_Layer(5, 2, sigma, learning_rate)
        self.Softmax_3 = Softmax_Activation()
        self.Net_4 = Matrix_Layer(2, 1, sigma, learning_rate)


    def forward_prediction(self, X):
        self.Net_1.forward_prediction(X)
        self.Softmax_1.forward_prediction(self.Net_1.outputs)
        self.Net_2.forward_prediction(self.Softmax_1.outputs)
        self.Softmax_2.forward_prediction(self.Net_2.outputs)
        self.Net_3.forward_prediction(self.Softmax_2.outputs)
        self.Softmax_3.forward_prediction(self.Net_3.outputs)
        self.Net_4.forward_prediction(self.Softmax_3.outputs)
        self.outputs = self.Net_4.outputs
    def get_entropy_variables(self, y, y_pred, i, j):
        '''
        :param i: the index of the first selected row
        :param j: the index of the second selected row
        :return:  the cross entropy variables S_{ij}, s_{i} and s_{j}associated with Row i and Row j
        '''
        s_i = y_pred[i][0] # s_i in Burges' paper, a sigmoid value from 0 to 1
        s_j = y_pred[j][0] # s_j in Burges' paper, a sigmoid value from 0 to 1

        if y[i][0] > y[j][0]: S_i_j = 1
        elif y[i][0] < y[j][0]: S_i_j = -1
        else: S_i_j = 0

        return [S_i_j, s_i, s_j]
    def cross_entropy_loss_one_pair(self, y, y_pred, i, j):
        '''
        :param i: the index of the first selected row
        :param j: the index of the second selected row
        :return:  the cross entropy associated with Row i and Row j
        '''
        [S_i_j, s_i, s_j] = self.get_entropy_variables(y, y_pred, i, j)
        C = (1 / 2) * (1 - S_i_j) * self.sigma * (s_i - s_j) \
            + np.log(1 + np.exp(-self.sigma * (s_i - s_j)))
        return C

    def cross_entropy_loss_total(self, y, y_pred):
        C = 0
        for i in np.arange(0, len(y) - 1):
            for j in np.arange(i + 1, len(y)):
                C += self.cross_entropy_loss_one_pair(y, y_pred, i, j)
        return C

    def cross_entropy_derivative_with_respect_to_s(self, y, y_pred, i, j):
        '''
        :return: the derivative of C with respect to s_{i}, also equivalent to
                 the negative of the derivative of C with respect to s_{j}
        '''
        [S_i_j, s_i, s_j] = self.get_entropy_variables(y, y_pred, i, j)
        return self.sigma * (0.5 * (1 - S_i_j) - 1 / (1 + np.exp(self.sigma * (s_i - s_j))))

    def relevance_score_derivative_with_respect_to_b_final(self, j):
        '''
        :param j: corresponds to s_j, the j_th sample
        :return: ds_j/db, where s_j = XW+b, X is output of previous layer
        '''
        #outputs_j = self.outputs[j][0]
        #ds_j_over_dlast_net_inputs = outputs_j
        return 1

    def relevance_score_derivative_with_respect_to_w_final(self, j, k):
        '''
        :param j: index of s (pred_relevance), the jth sample corresponds to s_j
        :param k: index of weights
        :return: ds_{j}/dw_{k}
        '''

        ds_j_over_dlast_net_inputs = self.relevance_score_derivative_with_respect_to_b_final(j)
        ds_j_over_dw_k = ds_j_over_dlast_net_inputs * self.Net_4.inputs[j][k]
        return ds_j_over_dw_k

    def relevance_score_derivative_with_respect_to_inputs_final(self, j, l, k):
        '''
        :param j: sample index of samples, s_{j} corresponds to the i_th sample
        :param l: row index of sigmoid input matrix
        :param k: column index of sigmoid input matrix
        :return: ds_{j}/dinputs_{l,k}
        '''

        ds_j_over_dlast_net_inputs = self.relevance_score_derivative_with_respect_to_b_final(j)
        ds_j_over_dinputs_l_k = int(j==l) * ds_j_over_dlast_net_inputs * self.Net_4.weights[k][0]
        return ds_j_over_dinputs_l_k


    def cross_entropy_derivative_with_respect_to_w_final(self, y, y_pred, k):
        '''
        :param k: index of weights
        :return: dC/dw_{k}
        '''
        dC_over_dw_k = 0
        ds_over_dw_k = [self.relevance_score_derivative_with_respect_to_w_final(i, k) \
                        for i in np.arange(0, len(self.Net_4.inputs))]  # avoid repeated calculation inside forloop

        for i in np.arange(0, len(self.Net_4.inputs) - 1):
            for j in np.arange(i + 1, len(self.Net_4.inputs)):
                dC_over_ds_i = self.cross_entropy_derivative_with_respect_to_s(y, y_pred, i, j)
                dC_over_ds_j = - dC_over_ds_i
                ds_i_over_dw_k = ds_over_dw_k[i]
                ds_j_over_dw_k = ds_over_dw_k[j]
                dC_over_dw_k = dC_over_dw_k + dC_over_ds_i * ds_i_over_dw_k + dC_over_ds_j * ds_j_over_dw_k

        return dC_over_dw_k


    def cross_entropy_derivative_with_respect_to_b_final(self, y, y_pred):
        '''
        :param k: index of weights
        :return: dC/dw_{k}
        '''
        dC_over_db = 0
        ds_over_db = [self.relevance_score_derivative_with_respect_to_b_final(i) \
                        for i in np.arange(0, len(y_pred))]  # avoid repeated calculation inside forloop

        for i in np.arange(0, len(y_pred) - 1):
            for j in np.arange(i + 1, len(y_pred)):
                dC_over_ds_i = self.cross_entropy_derivative_with_respect_to_s(y, y_pred, i, j)
                dC_over_ds_j = - dC_over_ds_i
                ds_i_over_db = ds_over_db[i]
                ds_j_over_db = ds_over_db[j]
                dC_over_db = dC_over_db + dC_over_ds_i * ds_i_over_db + dC_over_ds_j * ds_j_over_db

        return dC_over_db

    def cross_entropy_derivative_with_respect_to_inputs_final(self, y, y_pred, l, k):
        '''
        :param l: row index of sigmoid inputs
        :param k: column index of sigmoid inputs
        :return: dC/dsigmoid_inputs_{l,k}
        '''

        dC_over_dinputs_l_k = 0
        ds_over_dinputs_l_k = [self.relevance_score_derivative_with_respect_to_inputs_final(i, l, k) \
                                for i in np.arange(0, len(y_pred))]

        for i in np.arange(0, len(y_pred) - 1):
            for j in np.arange(i + 1, len(y_pred)):
                dC_over_ds_i = self.cross_entropy_derivative_with_respect_to_s(y, y_pred, i, j)
                dC_over_ds_j = - dC_over_ds_i
                ds_i_over_dinputs_l_k = ds_over_dinputs_l_k[i]
                ds_j_over_dinputs_l_k = ds_over_dinputs_l_k[j]
                dC_over_dinputs_l_k = dC_over_dinputs_l_k + dC_over_ds_i * ds_i_over_dinputs_l_k\
                                        + dC_over_ds_j * ds_j_over_dinputs_l_k

        return dC_over_dinputs_l_k

    def cross_entropy_derivative_with_respect_to_w_final_all(self, y, y_pred):
        return np.array([self.cross_entropy_derivative_with_respect_to_w_final(y, y_pred, k)
                for k in range(self.Net_4.inputs.shape[1])]).reshape(-1,1)

    def cross_entropy_derivative_with_respect_to_inputs_final_all(self, y, y_pred):
        dC_over_dNet_4_inputs = np.empty(shape=(1, self.Net_4.inputs.shape[1]))
        for row_index in range(Model.Net_4.inputs.shape[0]):
            new_row = [[self.cross_entropy_derivative_with_respect_to_inputs_final(y, y_pred,\
                        row_index, column_index) for column_index in range(self.Net_4.inputs.shape[1])]]
            dC_over_dNet_4_inputs = np.vstack((dC_over_dNet_4_inputs, new_row))
        return dC_over_dNet_4_inputs[1:, ]

    def one_round_of_forward_backward(self):
        # forward prediction
        self.forward_prediction(self.X_train)
        y_pred = self.outputs

        # backward propagation
        self.Net_4.bias -= self.cross_entropy_derivative_with_respect_to_b_final(self.y_train, y_pred)
        self.Net_4.weights -= self.cross_entropy_derivative_with_respect_to_w_final_all(self.y_train, y_pred)
        self.Net_4.dinputs = self.cross_entropy_derivative_with_respect_to_inputs_final_all(self.y_train, y_pred)
        self.Softmax_3.backward_propagation(self.Net_4.dinputs)
        self.Net_3.backward_propagation(self.Softmax_3.dinputs)
        self.Softmax_2.backward_propagation(self.Net_3.dinputs)
        self.Net_2.backward_propagation(self.Softmax_2.dinputs)
        self.Softmax_1.backward_propagation(self.Net_2.dinputs)
        self.Net_1.backward_propagation(self.Softmax_1.dinputs)



'''
Construct Data Set, X stores the features, y stores the final relevance scores
'''
def abs_sin_sum_square(num_list):
    return np.abs(np.sin(np.sum([item**2 for index, item in enumerate(num_list)])))

X_train = np.random.random((50, 5))
y_train = np.array([abs_sin_sum_square(X_train[i,:]) for i in range(X_train.shape[0])]).reshape(-1,1)
X_test = np.random.random((5, 5))
y_test = np.array([abs_sin_sum_square(X_test[i,:]) for i in range(X_test.shape[0])]).reshape(-1,1)



'''
Train the RankNet using the training data.
We shall see the cross entropy loss drops as the number of iterations increases.
'''
sigma = 2
learning_rate = 0.0001 # 0.0001
iteration_rounds = 10000# 10000
Model = Neural_Network(X_train, y_train, X_test, y_test, sigma, learning_rate)

# Initialize the model with one forward prediction
Model.forward_prediction(X_train)

# Training the model with gradient descent of cross entropy
for round_number in range(iteration_rounds):
    Model.one_round_of_forward_backward()
    print(Model.cross_entropy_loss_total(y_train, Model.outputs))

'''
Evaluation of the trained model by test data
'''

y_test_ranking = Get_ranking_from_scores.rank_scores(y_test.reshape(-1))
y_test_ranking = y_test_ranking.rename(columns = {'Id':'Real_Ranking'}).reset_index()

Model.forward_prediction(X_test)
y_pred = Model.outputs


y_pred_ranking = Get_ranking_from_scores.rank_scores(y_pred.reshape(-1))
y_pred_ranking = y_pred_ranking.rename(columns = {'Id':'Pred_Ranking'}).reset_index()

Real_pred_ranking_comparison = pd.concat([y_test_ranking['Real_Ranking'],\
                                    y_pred_ranking['Pred_Ranking']], axis = 1)

def switch_to_int(number):
    return int(number)
Real_pred_ranking_comparison['Real_Ranking'] = Real_pred_ranking_comparison['Real_Ranking'].apply(switch_to_int)
Real_pred_ranking_comparison['Pred_Ranking'] = Real_pred_ranking_comparison['Pred_Ranking'].apply(switch_to_int)

# Display results
print(y_test)
print(y_pred)
print(Real_pred_ranking_comparison)


