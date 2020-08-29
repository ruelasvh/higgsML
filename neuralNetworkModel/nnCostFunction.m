% Based on Andrew Ng's Machine Learning Online Class
%

function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for the 2 layer neural network.
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Number of examples
m = size(X, 1);
         
% Initialize cost and gradiants 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% 
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J.
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. Return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, check
%         that the implementation is correct by running checkNNGradients.
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. This vector mapped into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%

% Each row of y_matrix is a row vector representing y(i) = {1...10}.
% For instance, for the first training example i = 1, if  y(1) = 4, 
% then y_matrix(i, :) is represented as y_matrix(1, :) = [0 0 0 1 0 0 0 0 0 0].
% Note, the digit 4 is encoded as the index of the column of y_matrix(i, :) that 
% is equal to 1. Reference page 5 of ex4.pdf.
y_matrix = eye(num_labels)(y, :);

% Compute h_theta(x) = a3 by feedforward propagation
a1 = [ones(m, 1) X]; % add the extra bias unit
z2 = a1 * Theta1';
a2 = [ones(m, 1) sigmoid(z2)]; % add the extra bias unit
z3 = a2 * Theta2';
a3 = sigmoid(z3); % essentially h_theta(x) for all training examples

% Compute unregularized cost matrix representing h_theta(x) for each class, i.e. 1-10.
% Specifically, each row of J_matrix is a row vector for the digit we are tying to classify. 
% So if, J_matrix(1, :) = [0 0 0.001 0 0.002 0 0 0 0.001 0], J_matrix(1, :) is the activation 
% of the k-th (depending on the corresponding y label) output unit for the first training example. 
% J_matrix size is m x num_labels.
J_matrix= ((-y_matrix) .* log(a3) - (1.0 - y_matrix) .* log(1.0 - a3));

% Vectorized form of the double sum over all the training examples, m, and all the labels, num_labels.
% In the literature, num_labels is K and m is i, so here K = 10 and i is 5000.
J_unreg = (1.0 / m) * sum(sum(J_matrix));

% Remove extra bias unit
Theta1(:, [1]) = [];
Theta2(:, [1]) = [];
J = J_unreg + (lambda/(2*m)) * (sum(sum(Theta1.^2)) + sum(sum(Theta2.^2)));

% Remove extra bias unit from nn_params, a more generalized approach, though it doesn't
% pass in the submit script.
% unrolled_Theta = nn_params;
% unrolled_Theta(1:hidden_layer_size, :) = [];
% unrolled_Theta(hidden_layer_size * (input_layer_size):hidden_layer_size * (input_layer_size) + num_labels, :) = [];
% J = J_unreg + (lambda/(2*m)) * sum(unrolled_Theta.^2);

% Compute gradients of Theta using vectorized backpropagation 
% Equations found in ex4.pdf page 9
delta3 = a3 - y_matrix;
delta2 = delta3 * Theta2 .* sigmoidGradient(z2);
D_1 = delta2' * a1;
D_2 = delta3' * a2;

Theta1_grad = D_1 ./ m;
Theta2_grad = D_2 ./ m;

% For j = 0 there is no regularization term, specifically, the bias term should not
% be regularized. This is represented by adding a column of zeros to Theta1 and Theta2.
regularization_Theta1_grad = (lambda/m) .* [zeros(size(Theta1, 1),1) Theta1];
regularization_Theta2_grad = (lambda/m) .* [zeros(size(Theta2, 1),1) Theta2];

Theta1_grad = Theta1_grad +  regularization_Theta1_grad;
Theta2_grad = Theta2_grad + regularization_Theta2_grad;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
