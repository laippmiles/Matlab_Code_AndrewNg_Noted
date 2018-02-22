function [h, display_array] = displayData(X, example_width)
%DISPLAYDATA Display 2D data in a nice grid
%   [h, display_array] = DISPLAYDATA(X, example_width) displays 2D data
%   stored in X in a nice grid. It returns the figure handle h and the 
%   displayed array if requested.

% Set example_width automatically if not passed in
%��� ����Ҫ��� ��example_width�� ��һ����������ڵĻ�����ѡ��X����������ƽ����������example_width.
if ~exist('example_width', 'var') || isempty(example_width) 
	example_width = round(sqrt(size(X, 2)));
    %��ʱ�õ�example_width = 20
end

% Gray Image
%��ɫ������Ч
%��Ȼ����winter��summer���������ѡ��r(�s���t)�q
colormap(gray);

% Compute rows, cols
[m,n] = size(X);
example_height = (n / example_width);
% example_height = 20; 

% Compute number of items to display
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);
%�������10*10

% Between images padding
% ����� pad��ʵ���ǻ���Ϊ�˷ֿ�ÿ��ͼ�����ó� 0 �Ա��¾������ˡ�
pad = 1;

% Setup blank display
display_array = - ones(pad + display_rows * (example_height + pad), ...
                       pad + display_cols * (example_width + pad));
% display_array ��һ�� ��1 + 10 * �� 20 + 1����1 + 10 * ��20 + 1�����ľ���
%����Ϊ����������
% Copy each example into a patch on the display array
curr_ex = 1;
for j = 1:display_rows
	for i = 1:display_cols
		if curr_ex > m, 
			break; 
		end
		% Copy the patch
		
		% Get the max value of the patch
		max_val = max(abs(X(curr_ex, :)));
		display_array(pad + (j - 1) * (example_height + pad) + (1:example_height), ...
		              pad + (i - 1) * (example_width + pad) + (1:example_width)) = ...
						reshape(X(curr_ex, :), example_height, example_width) / max_val;
        % ��ԭ�ȵ� X ����ÿ�б�׼���Ժ� 20 * 20 ����ʽ ��� display_array �е�Ԫ�ء�
        %�漰ͼ�����Ķ����˨r(�s���t)�q
		curr_ex = curr_ex + 1;
	end
	if curr_ex > m, 
		break; 
	end
end

% Display Image
h = imagesc(display_array, [-1 1]);

% Do not show axis
axis image off

drawnow;

end