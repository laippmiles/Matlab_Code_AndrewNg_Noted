function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.
%��ex1��featureNormalize������ͬ��������Matlab�Ĺ������Ч��
mu = mean(X);
X_norm = bsxfun(@minus, X, mu);
%bsxfun�����ǶԾ�����ÿ��Ԫ�ؽ��в����ĺ�����������ʽ��C = bsxfun(fun,A,B)��fun�Ǵ���Ҫִ�е�����
%�ص���ÿ��Ԫ��
%�ο���
%http://blog.csdn.net/ddreaming/article/details/52176790
sigma = std(X_norm);
%std�������׼ƫ��
X_norm = bsxfun(@rdivide, X_norm, sigma);


% ============================================================

end
