%% Machine Learning Online Class
%  Exercise 6 | Support Vector Machines
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% =============== Part 1: Loading and Visualizing Data ================
%  We start the exercise by first loading and visualizing the dataset. 
%  The following code will load the dataset into your environment and plot
%  the data.
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data1: 
% You will have X, y in your environment
Data1 = load('ex6data1.csv');
Data1_X = Data1(:,2:size(Data1,2));
Data1_y = Data1(:,1);
% Plot training data
plotData(Data1_X, Data1_y);
%显示数据，没啥看头

%% ==================== Part 2: Training Linear SVM ====================
%用Data1试用原始SVM
%  The following code will train a linear SVM on the dataset and plot the
%  decision boundary learned.
%

% Load from ex6data1: 
% You will have X, y in your environment

fprintf('\nTraining Linear SVM ...\n')

% You should try to change the C value below and see how the decision
% boundary varies (e.g., try C = 1000)
C = 100;
model = svmTrain(Data1_X, Data1_y, C, @linearKernel, 1e-3, 20);
%原始SVM用线性核函数直接内积即可，不用多写什么
%train里头有个func2str了解一下
visualizeBoundaryLinear(Data1_X, Data1_y, model);
model

%% =============== Part 3: Implementing Gaussian Kernel ===============
%  You will now implement the Gaussian kernel to use
%  with the SVM. You should complete the code in gaussianKernel.m
%试用高斯核函数
fprintf('\nEvaluating the Gaussian Kernel ...\n')

x1 = [1 2 1]; x2 = [0 4 -1]; sigma = 2;
sim = gaussianKernel(x1, x2, sigma);

fprintf(['Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :' ...
         '\n\t%f\n(this value should be about 0.324652)\n'], sim);

%% =============== Part 4: Visualizing Dataset 2 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')
%和Part1一样，先展示数据
% Load from ex6data2: 
% You will have X, y in your environment
Data2 = load('ex6data2.csv');
Data2_X = Data2(:,2:size(Data2,2));
Data2_y = Data2(:,1);
% Plot training data
figure(2);
plotData(Data2_X, Data2_y);

%% ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
%  After you have implemented the kernel, we can now use it to train the 
%  SVM classifier.
% 
fprintf('\nTraining SVM with RBF Kernel (this may take 1 to 2 minutes) ...\n');

% Load from ex6data2: 
% You will have X, y in your environment

% SVM Parameters
C = 1; sigma = 0.1;

% We set the tolerance and max_passes lower here so that the code will run
% faster. However, in practice, you will want to run the training to
% convergence.
%下一行用了匿名函数的写法
%这样就能用gaussianKernel了
model2= svmTrain(Data2_X, Data2_y, C, @(x1, x2) gaussianKernel(x1, x2, sigma)); 
visualizeBoundary(Data2_X, Data2_y, model2);
model2

%% =============== Part 6: Visualizing Dataset 3 ================
%  The following code will load the next dataset into your environment and 
%  plot the data. 
%

fprintf('Loading and Visualizing Data ...\n')

% Load from ex6data3: 
% You will have X, y in your environment
Data3 = load('ex6data3_train.csv');
Data3_X = Data3(:,2:size(Data3,2));
Data3_y = Data3(:,1);

figure(3);
% Plot training data
plotData(Data3_X, Data3_y);

%% ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

%  This is a different dataset that you can use to experiment with. Try
%  different values of C and sigma here.
% 

% Load from ex6data3: 
% You will have X, y in your environment
Data3_val = load('ex6data3_val.csv');
Data3_X_val = Data3_val(:,2:size(Data3_val,2));
Data3_y_val = Data3_val(:,1);

% Try different SVM Parameters here
[C, sigma] = dataset3Params(Data3_X, Data3_y, Data3_X_val, Data3_y_val);
%网格法，寻找最优C和sigma
% Train the SVM
model3= svmTrain(Data3_X, Data3_y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
model3
visualizeBoundary(Data3_X, Data3_y, model3);
%如果优化参数失效，分类面是画不出来的
%用的等高线方法画的图