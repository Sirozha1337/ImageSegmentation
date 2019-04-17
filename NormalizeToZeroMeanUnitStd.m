% Ќормализует данные
% A - матрица размером NxP
% где N - количество вокселей, P - количество timeseries
function [A] = NormalizeToZeroMeanUnitStd(A)
    % Ќормализаци€ к mean=0, std=1
    A = (A - mean(A(:))) ./ std(A(:));