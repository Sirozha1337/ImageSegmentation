%%  Converts image to imagesc format and writes to disk
%---input---------------------------------------------------------
%   image: image to save
%   path: path to image
%---output--------------------------------------------------------
%   colored_image: saved image 
function [colored_image] = SaveImage(image, path)
colored_image = ind2rgb(im2uint8(mat2gray(image)), parula(256));
imwrite(colored_image, path);