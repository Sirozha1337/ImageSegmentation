% ����������� ������ �� �����
% A - ������� �������� NxP
% ��� N - ���������� ��������, P - ���������� timeseries
function [A] = NormalizeToUnitLength(A)
    %A = NormalizeToZeroMeanUnitStd(A);
    for row=1:size(A,1)
        len = sqrt(sum(A(row, :) .^ 2));
        if len > 0
            A(row, :) = A(row, :)/len;
        end
    end