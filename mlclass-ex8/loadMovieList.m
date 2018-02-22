function movieList = loadMovieList(num_movies)
%GETMOVIELIST reads the fixed movie list in movie.txt and returns a
%cell array of the words
%   movieList = GETMOVIELIST() reads the fixed movie list in movie.txt 
%   and returns a cell array of the words in movieList.


%% Read the fixed movieulary list
fid = fopen('movie_ids.txt');

% Store all movies in cell array movie{}
n = num_movies;  % Total number of movies 

movieList = cell(n, 1);
for i = 1:n
    % Read line
    line = fgets(fid);
    %�Ի��з�Ϊ�磬�õ�ÿһ�е��ַ���
    % Word Index (can ignore since it will be = i)
    [~, movieName] = strtok(line, ' ');
    %��һ���ո�ǰ���ַ��������ǵ�Ӱ�ı�ţ�����ַ��������ǵ�Ӱ����Ϣ
    %ֻȡ�ո������ݼ���
    % Actual Word
    movieList{i} = strtrim(movieName);
end
fclose(fid);

end
