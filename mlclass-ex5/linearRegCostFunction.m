function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
NumOfData = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
%初始化J与grad
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%笔记7-3的正则化回归
h = X*theta;
hError = h - y;
sumSquaredError = sum(hError .^ 2);
regTermLeft = (1/(2 * NumOfData)) * sumSquaredError;

thetaWithoutBias = theta(2:end);
sumSquaredTheta = sum(thetaWithoutBias .^ 2);
regTermRight = (lambda / (2 * NumOfData)) * sumSquaredTheta;

J = regTermLeft + regTermRight;

grad = (1/NumOfData) * X' * hError;
%这里向量化用了转置
grad(2:end) = grad(2:end) + (lambda / NumOfData) * thetaWithoutBias;











% =========================================================================

grad = grad(:);

end
