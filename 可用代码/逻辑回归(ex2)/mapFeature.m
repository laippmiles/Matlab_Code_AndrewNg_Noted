function out = mapFeature(X1, X2)
% MAPFEATURE Feature mapping function to polynomial features
%
%   MAPFEATURE(X1, X2) maps the two input features
%   to quadratic features used in the regularization exercise.
%
%   Returns a new feature array with more features, comprising of 
%   X1, X2, X1.^2, X2.^2, X1*X2, X1*X2.^2, etc..
%
%   Inputs X1, X2 must be the same size
%
degree = 6;
out = ones(size(X1(:,1)));
%out = ones(size(X1(:,1)),size(X1(:,1))+degree);
for i = 1:degree
    for j = 0:i
        %out(:, end+1) = (X1.^(i-j)).*(X2.^j);
        %这两句含义相同
        out = [out (X1.^(i-j)).*(X2.^j)];
        %matlab运算的实质是矩阵运算，所以当让两个矩阵相乘时，是按矩阵相乘算出的，点乘则是相应位置的元素乘相应位置的元素。
    end
end