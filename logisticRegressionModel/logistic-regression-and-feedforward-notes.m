function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

z = X * theta; % define hypothesis parameter
H = sigmoid(z); % compute sigmoid, given by 1 / (1 - exp(-z))
E = H - y; % compute error
J_unreg = (1.0 / m) * (-(y)' * log(H) - (1.0 - y)' * log(1.0 - H)); % compute unregularized cost

grad_unreg = (1.0 / m) .* (X' * E);
theta_unbias = theta;
theta_unbias(1) = 0; % set bias weight to 0, i.e. parameter theta_0 should not be regularized

J = J_unreg + lambda * sum(theta_unbias .^ 2) / (2.0 * m); % compute regularized cost
grad = grad_unreg .+ (lambda/m) * theta_unbias;

grad = grad(:);

end
