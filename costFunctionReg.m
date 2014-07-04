function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

    % Initialize some useful values
    m = length(y); % number of training examples

    % You need to return the following variables correctly 

    grad = zeros(size(theta));
    
    %Calling the non regularised cost Function
    [J_star, grad_star] = costFunction(theta, X, y);

    %Adding the regularisation to the cost
    J = J_star + lambda/(2*m)*sum(theta(2:end));

    %Regularising the gradient. For j = 0
    grad(1) = grad_star(1);
    
    %Regularising the gradient for all other j
    for i = 2:length(theta)
        grad(i) = grad_star(i) + (lambda/m)* theta(i);
    end



% =============================================================

end
