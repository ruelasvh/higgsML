% Based on Andrew Ng's Machine Learning Online Class - Exercise 4 Neural Network Learning
%

% load data excluding headers, which is the first row
% X = csvread('training.csv', 1, 0)
% [d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,d33] = textread("training.csv", "%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d,%f,%f,%f,%f,%f,%f,%f,%f,%s", "headerlines",1)

% save data to matrix in a file, and when data is loaded it will be available as X
% save -binary training.mat X

%% Initialization
addpath(genpath("~/higgsML", ".git"));
clear ; close all; clc


%% ================ Initializing Pameters ================
%  In this part of the exercise, you will be starting to implment a two
%  layer neural network that classifies digits. You will start by
%  implementing a function to initialize the weights of the neural network
%  (randInitializeWeights.m)

% Load Training Data
fprintf('Loading the data ...\n');
load('training_der.mat');
load('training_pri.mat');
load('y.mat');

X = [X_der X_pri];

fprintf('\nDimension of X \n');
disp(size(X));

[m, n] = size(X);

fprintf('\nRandomly select 10 data points to display.\n');
rand_m_indices = randperm(m);
rand_10_indices = rand_m_indices(1:10);
disp(X(rand_10_indices, :));

% Set the number of input units, hidden units, and output units
input_layer_size = n;
hidden_layer_size = 15; % could use some generalized size based on the input layer size
num_labels = 2; % two labels, signal (s) and background (b)
fprintf('\nInput units %d \n', input_layer_size);
fprintf('Hidden units %d \n', hidden_layer_size);
fprintf('Output units %d \n', num_labels);

fprintf('\nProgram paused. Press enter to initialize NN params (Theta).\n');
pause;

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);
fprintf('\nTheta1 size\n');
disp(size(initial_Theta1));
fprintf('\nTheta2 size\n');
disp(size(initial_Theta2));

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nProgram paused. Press enter to check backpropagation.\n');
pause;

%% =============== Implement Backpropagation ===============
%  Once your cost matches up with ours, you should proceed to implement the
%  backpropagation algorithm for the neural network. You should add to the
%  code you've written in nnCostFunction.m to return the partial
%  derivatives of the parameters.
%
fprintf('\nChecking Backpropagation... \n');

%  Check gradients by running checkNNGradients
checkNNGradients;

fprintf('\nProgram paused. Press enter to check backpropagation w/ regularization.\n');
pause;


%% =============== Implement Regularization ===============
%  Once your backpropagation implementation is correct, you should now
%  continue to implement the regularization with the cost and gradient.
%

lambda = 3;
fprintf('\nChecking Backpropagation (w/ Regularization lambda = %d) ... \n', lambda);

checkNNGradients(lambda);
y = y .+ 1;

% Also output the costFunction debugging values
nn_params = [initial_Theta1(:) ; initial_Theta2(:)];
debug_J  = nnCostFunction(nn_params, input_layer_size, ...
                          hidden_layer_size, num_labels, X, y, lambda);

fprintf('\nCost at (fixed) debugging parameters (w/ lambda = %d): %f \n', lambda, debug_J);

fprintf('\nProgram paused. Press enter to train the neural net.\n');
pause;


%% =================== Training NN ===================
%  You have now implemented all the code necessary to train a neural 
%  network. To train your neural network, we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')

%  After you have completed the assignment, change the MaxIter to a larger
%  value to see how more training helps.
max_iter = 250;
options = optimset('MaxIter', max_iter);

%  You should also try different values of lambda
lambda = 4;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('\nProgram paused. Press enter to calculate model accuracy.\n');
pause;

%% ================= Implement Predict =================
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);
accuracy = mean(double(pred == y)) * 100;

fprintf('\nTraining Set Accuracy: %f \n', accuracy);

accuracyCsv = 'results/accuracy.csv';
A = csvread(accuracyCsv, 1, 0);
headers = {'id','accuracy','lambda','hidden_layer_size','max_iter','feat_size','cputime','timestamp'};
A = [A ; [A(end,1) + 1, accuracy, lambda, hidden_layer_size, max_iter, n, cputime(), time()]];
csvwrite(accuracyCsv, A, headers);

