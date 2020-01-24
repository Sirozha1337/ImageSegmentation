%%  Rotate a matrix
%---input---------------------------------------------------------
%
%
%---output--------------------------------------------------------
%
function [matrix] = RotateMatrix(matrix, dim)

sz = size(matrix);

for i=1:sz(dim)
    C = repmat({':'},1,numel(sz));
    C(dim) = {i};
    matrix(C{:}) = rot90(squeeze(matrix(C{:})));
end