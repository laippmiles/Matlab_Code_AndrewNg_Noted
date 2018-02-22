function visualizeFit(X, mu, sigma2)
%VISUALIZEFIT Visualize the dataset and its estimated distribution.
%   VISUALIZEFIT(X, p, mu, sigma2) This visualization shows you the 
%   probability density function of the Gaussian distribution. Each example
%   has a location (x1, x2) that depends on its feature values.
%

[X1,X2] = meshgrid(0:.5:35); 
%�˴���.5ָ����Ϊ0.5
%�ٸ����ӣ�
X_contour = [X1(:) X2(:)];
%{
[X1(:) X2(:)]�ɵõ�0:.5:35֮�����ȡֵ��������ϣ����˾��û������ɵģ���
A(:)�ɰѾ���A��Ԫ�ذ��е�˳���Ϊһ�У�����ת��Ϊ����
��A��һ������A=[3 4 2;1 5 3;4 7 1]
��A(:)Ϊ[3 1 4 4 5 7 2 3 1]
meshgrid��MATLAB�������������������ĺ���
%}
Zp = multivariateGaussian(X_contour,mu,sigma2);
%�����������0:.5:35֮��ʱp������ȡֵ
Zp = reshape(Zp,size(X1));
plot(X(:, 1), X(:, 2),'bx');
hold on;
% Do not plot if there are infinities
%ֻ��Zp������inf���ʱ�򻭵ȸ���
if (sum(isinf(Zp)) == 0)
    contour(X1, X2, Zp, 10.^(-20:3:0)');
end
hold off;

end