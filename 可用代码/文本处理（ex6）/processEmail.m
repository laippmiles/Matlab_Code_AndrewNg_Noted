function word_indices = processEmail(email_contents,vocabList)
%PROCESSEMAIL preprocesses a the body of an email and
%returns a list of word_indices 
%   word_indices = PROCESSEMAIL(email_contents) preprocesses 
%   the body of an email and returns a list of indices of the 
%   words contained in the email. 
%

% Load Vocabulary
% Init return value
word_indices = [];

% ========================== Preprocess Email ===========================

% Find the Headers ( \n\n and remove )
% Uncomment the following lines if you are working with raw emails with the
% full headers

% hdrstart = strfind(email_contents, ([char(10) char(10)]));
% email_contents = email_contents(hdrstart(1):end);
% Lower case
email_contents = lower(email_contents);
%下面会大量用到正则表达式
%regexp——用于对字符串进行查找，大小写敏感；
%regexpi——用于对字符串进行查找，大小写不敏感；
%regexprep——用于对字符串进行查找并替换。
% Strip all HTML
% Looks for any expression that starts with < and ends with > and replace
% and does not have any < or > in the tag it with a space
%除去<>里的所有内容，改为' '。如果有嵌套<>就只去除最内部那个<> 
email_contents = regexprep(email_contents, '<[^<>]+>', ' ');
%方括号：匹配括号内的任何一个
%^指：除后面提到的单字符外的所有字符
%除去用不上的符号
% Handle Numbers
% Look for one or more characters between 0-9
%将阿拉伯数字全换成单词"Number"
email_contents = regexprep(email_contents, '[0-9]+', 'number');
%将网站全换成单词"httpaddr" 
% Handle URLS
% Look for strings starting with http:// or https://
%\s指空白符，(http|https)://[^\s]*指以http:// 或 https://开头，匹配至第一个空白符为止
email_contents = regexprep(email_contents, ...
                           '(http|https)://[^\s]*', 'httpaddr');

% Handle Email Addresses
% Look for strings with @ in the middle
%匹配@前后除去空格外的左右字符，替换为单词'emailaddr'
email_contents = regexprep(email_contents, '[^\s]+@[^\s]+', 'emailaddr');

% Handle $ sign
%除去$符号，如果$后面跟着多个$也可以除去。除去后在原位置替换为单词'dollar'
email_contents = regexprep(email_contents, '[$]+', 'dollar');


% ========================== Tokenize Email ===========================
% Output the email to screen as well
fprintf('\n==== Processed Email ====\n\n');

% Process file
l = 0;

while ~isempty(email_contents)
    % Tokenize and also get rid of any punctuation
    %分词
    [str, email_contents] = ...
       strtok(email_contents, ...
              [' @$/#.-:&*+=[]?!(){},''">_<;%' char(10) char(13)]);
    %分词间隔如上行单引号所示，char(10) char(13)分别表示换行和回车
    % Remove any non alphanumeric characters
    %除去所有除去字母数字的字符
    str = regexprep(str, '[^a-zA-Z0-9]', '');

    % Stem the word 
    % (the porterStemmer sometimes has issues, so we use a try catch block)
    try str = porterStemmer(strtrim(str)); 
        catch
            str = '';
            continue;
    end;
    %去掉分词失败的单词
    % Skip the word if it is too short
    if length(str) < 1
       continue;
    end

    % Look up the word in the dictionary and add to word_indices if
    % found
    % ====================== YOUR CODE HERE ======================
    % Instructions: Fill in this function to add the index of str to
    %               word_indices if it is in the vocabulary. At this point
    %               of the code, you have a stemmed word from the email in
    %               the variable str. You should look up str in the
    %               vocabulary list (vocabList). If a match exists, you
    %               should add the index of the word to the word_indices
    %               vector. Concretely, if str = 'action', then you should
    %               look up the vocabulary list to find where in vocabList
    %               'action' appears. For example, if vocabList{18} =
    %               'action', then, you should add 18 to the word_indices 
    %               vector (e.g., word_indices = [word_indices ; 18]; ).
    % 
    % Note: vocabList{idx} returns a the word with index idx in the
    %       vocabulary list.
    % 
    % Note: You can use strcmp(str1, str2) to compare two strings (str1 and
    %       str2). It will return 1 only if the two strings are equivalent.
    %


    word_indices = [word_indices; find(ismember(vocabList, str))];
    %找到邮件中包含在词汇库内的单词，记下单词索引


    % =============================================================

    %底下是为了排版美观的
    % Print to screen, ensuring that the output lines are not too long
    if (l + length(str) + 1) > 78
        fprintf('\n');
        l = 0;
    end
    fprintf('%s ', str);
    l = l + length(str) + 1;

end

% Print footer
fprintf('\n\n=========================\n');

end
