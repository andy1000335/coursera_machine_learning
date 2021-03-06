function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


for i = 1:m
    h = 0;
    for j = 1:n
        h = h + theta(j, 1)*X(i, j);
    end
    h = sigmoid(h);
    J = J + (-y(i)*log(h)-(1-y(i))*log(1-h));
end
regular = 0;
for j = 2:n
    regular = regular + theta(j)^2;
end
J = J/m + regular*lambda/(2*m);

for k = 1:n
    gradient = 0;
    for i = 1:m
        h = 0;
        for j = 1:n
            h = h + theta(j, 1)*X(i, j);
        end
        h = sigmoid(h);
        gradient = gradient + (h - y(i)) * X(i, k);
    end
    if (k ==1)
        grad(k) = gradient / m;
    else
        grad(k) = gradient/m + theta(k)*(lambda/m);
    endif
end


% =============================================================

end
