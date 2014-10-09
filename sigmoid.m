function g = sigmoid(z)
    %SIGMOID Compute sigmoid functoon
    %J = SIGMOID(z) computes the sigmoid of z.

    %Computes the sigmoid of each value of z. Vectorised so that it will
    %work correctly on a matrix, vector or scalar.
    g = 1 ./ (1 + exp(-z));

end
