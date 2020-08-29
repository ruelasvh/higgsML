% Based on Andrew Ng's Machine Learning Online Class
%

function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Number of examples  and number of labels
m = size(X, 1);
num_labels = size(Theta2, 1);

% Initialize prediction to a zero vector
p = zeros(m, 1);

%
% Note: The max function returns the index of the max element, for more
%       information see 'help max'. If the examples are in rows, then,
%       max(A, [], 2) can be used to obtain the max for each row.
%

% The following implementation is taken from Figure 2, page 11 of ex3.pdf
a1 = [ones(m, 1) X]; % add column of 1's, the extra bias unit
z2 = a1 * Theta1'; % use Theta1 transposed
a2 = sigmoid(z2);
a2 = [ones(m, 1) a2]; % add the extra bias unit
z3 = a2 * Theta2'; % use Theta2 transposed
a3 = sigmoid(z3);
% Use the indices (p) of the max values for prediction
[v p] = max(a3, [], 2); % get the max values of all a3 rows into column vectors

end
