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
    %以换行符为界，得到每一行的字符串
    % Word Index (can ignore since it will be = i)
    [~, movieName] = strtok(line, ' ');
    %第一个空格前的字符串内容是电影的编号，后的字符串内容是电影的信息
    %只取空格后的内容即可
    % Actual Word
    movieList{i} = strtrim(movieName);
end
fclose(fid);

end
