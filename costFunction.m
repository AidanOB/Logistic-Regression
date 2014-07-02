function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

    % Initialize some useful values
    m = length(y); % number of training examples

	%Clearing and initialising the vector.
    grad = zeros(size(theta));


    g = sigmoid(X*theta);
    J = (1/m)*sum(-y.*log(g) - (1-y).*log(1-g));
    error = g - y;
    for i = 1:length(theta)
        grad(i) = (1/m) * sum(error .* X(:,i));
    end

    





% =============================================================

end
