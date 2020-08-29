% From on Andrew Ng's Machine Learning Online Class
%

function W = randInitializeWeights(L_in, L_out)
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections
%   W = RANDINITIALIZEWEIGHTS(L_in, L_out) randomly initializes the weights 
%   of a layer with L_in incoming connections and L_out outgoing 
%   connections. 
%
%   Note that W should be set to a matrix of size(L_out, 1 + L_in) as
%   the first column of W handles the "bias" terms
%

% You need to return the following variables correctly 
W = zeros(L_out, 1 + L_in);

%
% W is randomly initialized so that the symmetry is broken while
% training the neural network. The first column of W corresponds 
% to the parameters for the bias unit.
%


% Symmetry breaking - Randomly initialize the weights to small values
% For choosing epsilon_init, see footnotes in ex4.pdf page 7
epsilon_init = 0.12; % epsilon_init = sqrt(6) / sqrt(L_in + L_out)
W = rand(L_out, 1 + L_in) * 2 * epsilon_init - epsilon_init;

end
