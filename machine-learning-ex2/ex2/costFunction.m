function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

for k = 1:n
    gradient = 0;
    J = 0;
    for i = 1:m
        h = 0;
        for j = 1:n
            h = h + theta(j, 1)*X(i, j);
        end
        h = sigmoid(h);
        J = J + (-y(i)*log(h)-(1-y(i))*log(1-h));
        gradient = gradient + (h - y(i)) * X(i, k);
    end
    J = J / m;
    grad(k) = gradient / m ;
end



% =============================================================

end
