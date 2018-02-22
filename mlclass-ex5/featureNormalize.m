function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%与ex1的featureNormalize功能相同，套用了Matlab的工具提高效率
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
%bsxfun函数是对矩阵中每个元素进行操作的函数，函数形式是C = bsxfun(fun,A,B)，fun是代表要执行的运算
%重点在每个元素
%参考：
%http://blog.csdn.net/ddreaming/article/details/52176790
sigma = std(X_norm);
%std：计算标准偏差
X_norm = bsxfun(@rdivide, X_norm, sigma);


% ============================================================

end
