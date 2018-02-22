%% Machine Learning Online Class
%这套代码全程没用测试集
%  Exercise 5 | Regularized Linear Regression and Bias-Variance
%  ex5主要用来展示如何分析Bias-Variance曲线
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     linearRegCostFunction.m
%     learningCurve.m
%     validationCurve.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =========== Part 1: Loading and Visualizing Data =============
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

% Load Training Data
fprintf('载入样本.\n')

% Load from ex5data1: 
% You will have X, y, Xval, yval, Xtest, ytest in your environment
load ('ex5data1.mat');
%.mat里有训练集、验证集和标签集（1维，12/21/21个）
%低维小数据，就是展示用这样
% m = Number of examples
NumOfData = size(X, 1);
fprintf('载入结束.\n')
% Plot training data
fprintf('显示样本\n')
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
%先简单将样本显示在坐标系上
%pause;

%% =========== Part 2: Regularized Linear Regression Cost =============
%  You should now implement the cost function for regularized linear 
%  regression. 
%
%初始化参数
theta = [1 ; 1];
lambda = 1;
[J, grad] = linearRegCostFunction([ones(NumOfData, 1) X], y, theta, lambda);

fprintf(['Cost at theta = [1 ; 1]: %f '...
         '\n(this value should be about 303.993192)\n'], J);

%pause;

%% =========== Part 3: Regularized Linear Regression Gradient =============
%  You should now implement the gradient for regularized linear 
%  regression.
%
fprintf(['Gradient at theta = [1 ; 1]:  [%f; %f] '...
         '\n(this value should be about [-15.303016; 598.250744])\n'], ...
         grad(1), grad(2));

%pause;


%% =========== Part 4: Train Linear Regression =============
%  Once you have implemented the cost and gradient correctly, the
%  trainLinearReg function will use your cost function to train 
%  regularized linear regression.
% 
%  Write Up Note: The data is non-linear, so this will not give a great 
%                 fit.
%

%  Train linear regression with lambda = 1
%先训练一个没有正则项的LR
lambda = 1;
[theta] = trainLinearReg([ones(NumOfData, 1) X], y, lambda);

%  Plot fit over the data 
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(X, [ones(NumOfData, 1) X]*theta, '--', 'LineWidth', 2)
hold off;
%hold on 和hold off，是相对使用的
%前者的意思是，你在当前图的轴（坐标系）中画了一幅图，再画另一幅图时，原来的图还在，与新图共存，都看得到
%后者表达的是状态是hold off时,再画另一幅图，原来的图就看不到了，在轴上绘制的是新图，原图被替换了
%pause;


%% =========== Part 5: Learning Curve for Linear Regression =============
%  Next, you should implement the learningCurve function. 
%
%  Write Up Note: Since the model is underfitting the data, we expect to
%                 see a graph with "high bias" -- slide 8 in ML-advice.pdf 
%

lambda = 1;
[error_train, error_val] = ...
    learningCurve([ones(NumOfData, 1) X], y, ...
                  [ones(size(Xval, 1), 1) Xval], yval, ...
                  lambda);
figure(2);
plot(1:NumOfData, error_train, 1:NumOfData, error_val);
title('Learning curve for linear regression')
legend('Train', 'Cross Validation')
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 150])

fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:NumOfData
    fprintf('  \t%d\t\t%f\t%f\n', i, error_train(i), error_val(i));
end
%这部分是展示训练样本大小和训练效果的曲线图
%pause;

%% =========== Part 6: Feature Mapping for Polynomial Regression =============
%  One solution to this is to use polynomial regression. You should now
%  complete polyFeatures to map each example into its powers
%扩展特征
%多项式回归，p为多项式项数（也是扩增特征特征个数-1）（除x0外原特征为第一项）
p = 8;

% Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X, p);
[X_poly, mu, sigma] = featureNormalize(X_poly);  % Normalize归一化
X_poly = [ones(NumOfData, 1), X_poly];                   % Add Ones加入x0

% Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p);
[X_poly_test, ~, ~] = featureNormalize(X_poly_test);  % Normalize归一化
X_poly_test = [ones(size(X_poly_test, 1), 1), X_poly_test];         % Add Ones

% Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p);
[X_poly_val, ~, ~] = featureNormalize(X_poly_val);
X_poly_val = [ones(size(X_poly_val, 1), 1), X_poly_val];           % Add Ones

fprintf('Normalized Training Example 1:\n');
fprintf('  %f  \n', X_poly(1, :));

%pause;



%% =========== Part 7: Learning Curve for Polynomial Regression =============
%  Now, you will get to experiment with polynomial regression with multiple
%  values of lambda. The code below runs polynomial regression with 
%  lambda = 0. You should try running the code with different values of
%  lambda to see how the fit and learning curve change.
%
%用来对比扩增特征的效果
%生效了
lambda = 1;
[theta] = trainLinearReg(X_poly, y, lambda);

% Plot training data and fit
figure(3);
plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
plotFit(min(X), max(X), mu, sigma, theta, p);
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
title (sprintf('Polynomial Regression Fit (lambda = %f)', lambda));

figure(4);
[error, error_val] = ...
    learningCurve(X_poly, y, X_poly_val, yval, lambda);
plot(1:size(error,1), error, 1:size(error_val,1), error_val);

title(sprintf('Polynomial Regression Learning Curve (lambda = %f)', lambda));
xlabel('Number of training examples')
ylabel('Error')
axis([0 13 0 100])
legend('Test', 'Cross Validation')

fprintf('Polynomial Regression (lambda = %f)\n\n', lambda);
fprintf('# Training Examples\tTrain Error\tCross Validation Error\n');
for i = 1:NumOfData
    fprintf('  \t%d\t\t%f\t%f\n', i,error_val(i),error(i));
end
%格式：样本编号，训练误差，验证误差
%pause;

%% =========== Part 8: Validation for Selecting Lambda =============
%  You will now implement validationCurve to test various values of 
%  lambda on a validation set. You will then use this to select the
%  "best" lambda value.
%验证lambda取值对预测效果的影响

[lambda_vec, error_train, error_val] = ...
    validationCurve(X_poly, y, X_poly_val, yval);

figure(5);
plot(lambda_vec, error_train, lambda_vec, error_val);
legend('Train', 'Cross Validation');
xlabel('lambda');
ylabel('Error');

fprintf('lambda\t\tTrain Error\tValidation Error\n');
for i = 1:length(lambda_vec)
	fprintf(' %f\t%f\t%f\n', ...
            lambda_vec(i), error_train(i), error_val(i));
end

%pause;
