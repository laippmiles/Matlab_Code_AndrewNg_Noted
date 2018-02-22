%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%
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

%% =============== Part 1: Loading movie ratings dataset ================
%  You will start by loading the movie ratings dataset to understand the
%  structure of the data.
%  
fprintf('Loading movie ratings dataset.\n\n');

%  Load data
Y = load('ex8_movie_Train_Y.csv');
R =  logical(load('ex8_movie_Train_R.csv'));

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%  Y（i行j列）记载了用户j对电影i的评分
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i
%  R（i行j列）记载了用户j对电影i的评分情况(是否评分)
%  From the matrix, we can compute statistics like average rating.
%  统计第一部电影的平均得分
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
imagesc(Y);
%将Y的数据图像化（一群闪瞎人的点点点）
ylabel('Movies');
xlabel('Users');

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  You will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in 
%  cofiCostFunc.m to return J.

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
Theta = load('ex8_movieParams_Theta.csv');
X = load('ex8_movieParams_Train.csv');
num_movies = size(X,1);
num_users = size(Theta,1);
num_features = size(X,2);
%  Reduce the data set size so that this runs faster
%num_users = 4; num_movies = 5; num_features = 3;
%其实用不用齐数据好像差不多...╮(╯▽╰)╭？
%X = X(1:num_movies, 1:num_features);
%Theta = Theta(1:num_users, 1:num_features);
%Y = Y(1:num_movies, 1:num_users);
%R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
J = cofiCostFunc(X ,Theta, Y, R,0);
           
fprintf(['Cost at loaded parameters: %f '...
         '\n(（缩减数据后）this value should be about 22.22)\n'], J);

%% ============== Part 3: Collaborative Filtering Gradient ==============
%  Once your cost function matches up with ours, you should now implement 
%  the collaborative filtering gradient function. Specifically, you should 
%  complete the code in cofiCostFunc.m to return the grad argument.
%  
fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction;
%进行梯度检验，确认Cost函数是有效的
%详见笔记9-5

%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  

%  Evaluate cost function
J = cofiCostFunc(X ,Theta, Y, R,1.5);
%改变Lambda值，看Cost函数值变化          
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n'], J);

%% ======= Part 5: Collaborative Filtering Gradient Regularization ======
%  Once your cost matches up with ours, you should proceed to implement 
%  regularization for the gradient. 
%

%  
fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
checkCostFunction(1.5);
%再次进行梯度检验，此时Lambda值为1.5

%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
%先输入某客户先前对一些电影的评分，并显示
movieList = loadMovieList(num_movies);

%  Initialize my ratings
my_ratings = zeros(num_movies, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2;

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;
%打印输出新用户的评分情况
fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%
%这里开始用的协同过滤
fprintf('\nTraining collaborative filtering...\n');

%  Load data
%load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];
% 目前的训练样本只有Y和R，其中第一列是指等会输出的“新客户”预测投票情况
%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);
%这时X和Theta都为待预测值
%initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);
initial_parameters = [X(:); Theta(:)];
% Set Regularization
lambda = 10;
%自己改的CostFunc只是单纯的CostFunc，虽然直观但是对多变量支持不好
theta = fmincg (@(t)(collaborative_filtering_CostFunc(t, Y, R, num_users, num_movies, num_features, lambda)), ...
                initial_parameters, options);
% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);
%以上两个变量都是优化来的
fprintf('Recommender system learning completed.\n');

%% ================== Part 8: Recommendation for you ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%
%用16-4的协同过滤方法优化
p = X * Theta';
my_predictions = p(:,1) ;
movieList = loadMovieList(num_movies);

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

