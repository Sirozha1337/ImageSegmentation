% ����������� ������
% A - ������� �������� NxP
% ��� N - ���������� ��������, P - ���������� timeseries
function [A] = NormalizeToZeroMeanUnitStd(A)
    % ������������ � mean=0, std=1
    A = (A - mean(A(:))) ./ std(A(:));