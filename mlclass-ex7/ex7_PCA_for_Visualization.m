%% === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
%  One useful application of PCA is to use it to visualize high-dimensional
%  data. In the last K-Means exercise you ran K-Means on 3-dimensional 
%  pixel colors of an image. We first visualize this output in 3D, and then
%  apply PCA to obtain a visualization in 2D.

close all; close all; clc

% Re-load the image from the previous exercise and run K-Means on it
% For this to work, you need to complete the K-Means assignment first
A = double(imread('bird_small.png'));

% If imread does not work for you, you can try instead
%   load ('bird_small.mat')
A = A / 255;
%上面这个必须要乘的
img_size = size(A);
X = reshape(A, img_size(1) * img_size(2), 3);
K = 2; 
max_iters = 10;
initial_centroids = kMeansInitCentroids(X, K);
[centroids, idx] = runkMeans(X, initial_centroids, max_iters);

%  Sample 1000 random indexes (since working with all the data is
%  too expensive. If you have a fast computer, you may increase this.
sel = floor(rand(1000, 1) * size(X, 1)) + 1;
%只随机展示1000个样本
%  Setup Color Palette
palette = hsv(K);
colors = palette(idx(sel), :);

%  Visualize the data and centroid memberships in 3D
figure(2);
scatter3(X(sel, 1), X(sel, 2), X(sel, 3), 10, colors);
title('Pixel dataset plotted in 3D. Color shows centroid memberships');
%先用K-mean把给标签标色

%% === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
% Use PCA to project this cloud to 2D for visualization

% Subtract the mean to use PCA
[X_norm, mu, sigma] = featureNormalize(X);
%X取值不受K-mean影响，K-mean可理解为用来给样本着色

% PCA and project the data to 2D
PCA_K = 2;
%降为2维样本
%不晓得为啥这个图只能降到二维以下...？
%因为每个像素点本来就只有三个特征...
[U, ~] = pca(X_norm);
Z = projectData(X_norm, U, PCA_K);
% Plot in 2D
figure(3);
plotDataPoints(Z(sel, :), idx(sel), K);
%只展示随机抽到那1000个样本的PCA情况
title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction');

X_recovered  = recoverData(Z, U, PCA_K);
X_recovered = reshape(X_recovered, img_size(1), img_size(2), 3);

figure(4);
% Display the original image 
subplot(1, 2, 1);
imagesc(A); 
%K-mean
title('After K-mean');

% Display compressed image side by side
subplot(1, 2, 2);
imagesc(X_recovered)
title(sprintf('After %d PCA', K));
%对比原始数据和解压后得到的近似数据组成的图像
%还原度还不错，会糊一点