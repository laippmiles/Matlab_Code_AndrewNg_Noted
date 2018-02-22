function visualizeFit(X, mu, sigma2)
%VISUALIZEFIT Visualize the dataset and its estimated distribution.
%   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
%   probability density function of the Gaussian distribution. Each example
%   has a location (x1, x2) that depends on its feature values.
%

[X1,X2] = meshgrid(0:.5:35); 
%此处的.5指步长为0.5
%举个例子：
X_contour = [X1(:) X2(:)];
%{
[X1(:) X2(:)]可得到0:.5:35之间可能取值的所有组合（个人觉得还蛮技巧的？）
A(:)可把矩阵A的元素按列的顺序变为一列，矩阵转化为向量
如A是一个矩阵，A=[3 4 2;1 5 3;4 7 1]
则A(:)为[3 1 4 4 5 7 2 3 1]
meshgrid是MATLAB中用于生成网格采样点的函数
%}
Zp = multivariateGaussian(X_contour,mu,sigma2);
%求出两特征在0:.5:35之间时p的所有取值
Zp = reshape(Zp,size(X1));
plot(X(:, 1), X(:, 2),'bx');
hold on;
% Do not plot if there are infinities
%只在Zp不存在inf点的时候画等高线
if (sum(isinf(Zp)) == 0)
    contour(X1, X2, Zp, 10.^(-20:3:0)');
end
hold off;

end