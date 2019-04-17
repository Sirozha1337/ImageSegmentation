% Нормализует данные по длине
% A - матрица размером NxP
% где N - количество вокселей, P - количество timeseries
function [A] = NormalizeToUnitLength(A)
    %A = NormalizeToZeroMeanUnitStd(A);
    for row=1:size(A,1)
        len = sqrt(sum(A(row, :) .^ 2));
        if len > 0
            A(row, :) = A(row, :)/len;
        end
    end