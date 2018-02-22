%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%  涉及高斯分布问题
%  异常检测问题示例
%  异常检测也称离群点检测
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easy to
%  visualize.
%
%  Our example case consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  This exercise will help us find possibly faulty (or very fast) machines.
%
%一如既往，先在坐标系展示数据
fprintf('Visualizing example dataset for outlier detection.\n\n');

%  The following command loads the dataset. You should now have the
%  variables X, Xval, yval in your environment
X = load('ex8data1_Sample.csv');

%  Visualize the example dataset
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');

%% ================== Part 2: Estimate the dataset statistics ===================
%  For this exercise, we assume a Gaussian distribution for the dataset.
%
%  We first estimate the parameters of our assumed Gaussian distribution, 
%  then compute the probabilities for each of the points and then visualize 
%  both the overall distribution and where each of the points falls in 
%  terms of that distribution.
%
fprintf('Visualizing Gaussian fit.\n\n');

%  Estimate mu and sigma2
[mu,sigma2] = estimateGaussian(X);
%先得到样本的均值mu和方差sigma2
%  Returns the density of the multivariate normal at each data point (row) 
%  of X
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
visualizeFit(X,  mu, sigma2);
%其实等高线的内容很微妙...好像不搞也能出最后的结果
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

%% ================== Part 3: Find Outliers ===================
%  Now you will find a good epsilon threshold using a cross-validation set
%  probabilities given the estimated Gaussian distribution
% 
%借助验证集（验证集有标签）来学习得到检测的阈值
Val = load('ex8data1_Val.csv');
Xval = Val(:,2:size(Val,2));
yval = Val(:,1);
pval = multivariateGaussian(Xval, mu, sigma2);

[epsilon,F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n\n');
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);


%  Find the outliers in the training set and plot the
outliers = find(p < epsilon);
%找到异常边界点
%  Draw a red circle around those outliers
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off


%% ================== Part 4: Multidimensional Outliers ===================
%  We will now use the code from the previous part and apply it to a 
%  harder problem in which more features describe each datapoint and only 
%  some features indicate whether a point is an outlier.
%

%  Loads the second dataset. You should now have the
%  variables X, Xval, yval in your environment
%data2与data1的差别在data2是多维的，不过因为多元高斯分布法本来就兼容多特征的分类，其实用的是一个执行方法。
fprintf('\n')
fprintf('---------Begin  Multidimensional Outliers detection.------------- \n');
X2 = load('ex8data2_Sample.csv');
Data2_Val = load('ex8data2_Val.csv');
X2_Val = Data2_Val(:,2:size(Data2_Val,2));
y2_Val = Data2_Val(:,1);
%  Apply the same steps to the larger dataset
[mu,sigma2] = estimateGaussian(X2);

%  Training set 
p2 = multivariateGaussian(X2, mu, sigma2);

%  Cross-validation set
p2val = multivariateGaussian(X2_Val, mu, sigma2);

%  Find the best threshold
[epsilon2,F1] = selectThreshold(y2_Val, p2val);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon2);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n\n');
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('# Outliers found: %d\n', sum(p2 < epsilon2));



