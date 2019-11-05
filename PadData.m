%%  Алгоритм нахождения MAP-оценки
%---input---------------------------------------------------------
%
%---output--------------------------------------------------------
%   
function [realigned_data]=PadData(data, prepadding, postpadding)

realigned_data = padarray(data, prepadding, 0, 'pre');
realigned_data = padarray(realigned_data, postpadding, 0, 'post');