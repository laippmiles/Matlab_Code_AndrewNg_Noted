function plotProgresskMeans(X, centroids, previous, idx, K, i)
%PLOTPROGRESSKMEANS is a helper function that displays the progress of 
%k-Means as it is running. It is intended for use only with 2D data.
%   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
%   points with colors assigned to each centroid. With the previous
%   centroids, it also plots a line between the previous locations and
%   current locations of the centroids.
%

% Plot the examples
% Create palette
palette = hsv(K+1);
%这个是调色彩空间的，K或者K+1都没事
colors = palette(idx, :);

% Plot the data
scatter(X(:,1), X(:,2), 15, colors);
%画出样本集的散点图
% Plot the centroids as black x's
plot(centroids(:,1), centroids(:,2), 'x', ...
     'MarkerEdgeColor','k', ...
     'MarkerSize', 10, 'LineWidth', 3);

% Plot the history of the centroids with lines
for j=1:size(centroids,1)
    drawLine(centroids(j, :), previous(j, :));
end
%连接当前和前一个聚类簇的两点
% Title
title(sprintf('Iteration number %d', i))

end

