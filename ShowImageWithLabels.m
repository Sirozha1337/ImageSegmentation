%%  Shows slice of an image overlayed with labels
%---input---------------------------------------------------------
%
%
%---output--------------------------------------------------------
%
function [] = ShowImageWithLabels(image, labels, direction, slice_num)
C = repmat({':'},1,ndims(image));
C(direction) = {slice_num};
slice = image(C{:});
slice = squeeze(mat2gray(slice));

C = repmat({':'},1,ndims(labels));
C(direction) = {slice_num};
labels_slice = squeeze(labels(C{:}));
overlayed_img = labeloverlay(slice,labels_slice);
imshow(overlayed_img);