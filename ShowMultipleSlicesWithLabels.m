%%  Shows multiple slices with labels
%---input---------------------------------------------------------
%
%
%---output--------------------------------------------------------
%
function [currentFigure] = ShowMultipleSlicesWithLabels(image, labels, directions, slice_nums, titles)

len = numel(directions);

figure;  
for i=1:len
    direction = directions(i);
    slice_num = slice_nums(i);
    subplot(len,1,i)
	ShowImageWithLabels(image, labels, direction, slice_num); 
end
currentFigure = gcf;
for i=1:len
    title(currentFigure.Children(len-i+1), titles(i));
end

