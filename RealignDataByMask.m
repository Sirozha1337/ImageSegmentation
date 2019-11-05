%%  Алгоритм нахождения MAP-оценки
%---input---------------------------------------------------------
%
%---output--------------------------------------------------------
%   
function [realigned_data, prepadding, postpadding]=RealignDataByMask(data, mask)

dim_num = ndims(mask);
offset = ones(dim_num,1);
rms = ones(1, dim_num) .* size(mask);

for i=1:dim_num
   for j=1:size(mask, i)
       C = repmat({':'},1,dim_num);
       C(i) = {j};
       slice = mask(C{:});
       if sum(slice.^2) == 0
           offset(i) = j+1;
       else
           break;
       end
   end
   for j=size(mask, i):-1:1
       C = repmat({':'},1,dim_num);
       C(i) = {j};
       slice = mask(C{:});
       if sum(slice.^2) == 0
           rms(i) = j-1;
       else
           break;
       end
   end
end
dim_num_data = ndims(mask);
if ndims(data) ~= ndims(mask)
    dim_num_data = ndims(data);
end
C = repmat({':'},1,dim_num_data);
prepadding = zeros(1, dim_num);
postpadding = zeros(1, dim_num);
for i=1:dim_num
    C(i) = {str2num(strcat(num2str(offset(i)), ':', num2str(rms(i))))}; %#ok<ST2NM>
    prepadding(i) = offset(i) - 1;
    postpadding(i) = size(mask, i) - rms(i);
end
if ndims(data) ~= ndims(mask)
    C(ndims(data)) = {':'};
end
realigned_data = data(C{:});
