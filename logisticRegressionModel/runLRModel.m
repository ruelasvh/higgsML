%% Based on Andrew Ng's Machine Learning Online Class - Exercise 2: Logistic Regression
%
%  Linear Regresion For Signal To Background In The Tau-Tau Channel
%  ------------
%
%  This file contains code which covers regularization with logistic regression.
%
%  Functions in this code:
%
%     sigmoid.m
%     predict.m
%     lrCostFunction.m
%
%

% Notes on loading and saving files
% load data excluding headers, which is the first row
% X = csvread('training.csv', 1, 0)
% [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,d33] = textread("training.csv", "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%f,%f,%f,%f,%f,%f,%f,%f,%s", "headerlines",1)

% save data to matrix in a file, and when data is loaded it will be available as X
% save -binary training.mat X

%% Initialization
addpath(genpath("~/higgsML", ".git"));
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%   
%

fprintf('Loading and Visualizing Data ...\n')

load('training_der.mat');
load('training_pri.mat');
load('y.mat');

X = [X_der X_pri];

fprintf('\nDimension of X \n');
disp(size(X));

[m, n] = size(X);

% Add intercept term to X
X = [ones(m, 1) X];

% % Randomly select 10 data points to display
fprintf('\nRandomly select 100 data points to display.\n');
rand_m_indices = randperm(m);
rand_100_indices = rand_m_indices(1:100);
% disp(X(rand_100_indices, :));

% Plot only two cols, 4 and 9, DER_pt_h and DER_pt_tot, respectively
plotData(X(rand_100_indices, [4,9]), y(rand_100_indices, :));

% Put some labels
hold on;

% Labels and Legend
xlabel('DER-pt-h')
ylabel('DER-pt-tot')

% Specified in plot order
h = legend ({"y = s", "y = b"}, "location", "northeastoutside");
set (h, "fontsize", 16);
legend right;

hold off;

fprintf('\nProgram paused. Press enter to test lrCostFunction.\n');
pause;


%% =========== Part 2: Regularized Logistic Regression ============
%  
%  

% Test case for lrCostFunction
fprintf('\nTesting lrCostFunction() with regularization');

theta_t = [-2; -1; 1; 2];
X_t = [ones(5,1) reshape(1:15,5,3)/10];
y_t = ([1;0;1;0;1] >= 0.5);
lambda_t = 3;
[J, grad] = lrCostFunction(theta_t, X_t, y_t, lambda_t);

fprintf('\nCost: %f\n', J);
fprintf('Expected cost: 2.534819\n');
fprintf('Gradients:\n');
fprintf(' %f \n', grad);
fprintf('Expected gradients:\n');
fprintf(' 0.146561\n -0.548558\n 0.724722\n 1.398003\n');

fprintf('\nProgram paused. Press enter to test using zero vector for theta.\n');
pause;

% Compute and display initial cost and gradient for regularized logistic
% regression
initial_theta = zeros(size(X,2), 1);
lambda = 1;
[J, grad] = lrCostFunction(initial_theta, X, y, lambda);

fprintf('\nCost at initial theta (zeros): %f\n', J);
fprintf('Expected cost (approx): 0.693\n');
fprintf('Gradient at initial theta (zeros) - first five values only:\n');
fprintf(' %f \n', grad(1:5));

fprintf('\nProgram paused. Press enter to train using fminunc.\n');
pause;

%% ============= Part 3: Regularization and Accuracies =============
%  In this part, try different values of lambda and
%  see how regularization affects the decision coundart
%
%  Try the following values of lambda (0, 1, 10, 100).
%
%  How does the decision boundary change when you vary lambda? How does
%  the training set accuracy vary?
%

% Train using fminunc
fprintf('\nTrain using fminunc\n');

% Initialize fitting parameters
initial_theta = zeros(size(X,2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 0;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(lrCostFunction(t, X, y, lambda)), initial_theta, options);

% % Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy (with lambda = %g): %f\n', lambda, mean(double(p == y)) * 100);

fprintf('\nProgram paused. Press enter to plot decision boundry.\n');
pause;

% Plot decision boundry
fprintf('\nPlotting decision boundry\n');

plotDecisionBoundary(theta([1,4,9]), X(rand_100_indices, [1,4,9]), y(rand_100_indices, :));
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend
xlabel('DER-pt-h')
ylabel('DER-pt-tot')

hold off;
