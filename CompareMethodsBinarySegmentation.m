
image = imread('data/a_8049.jpg');
gt_image = imread('data/a_8049_gt.png');
gt = zeros([size(gt_image,1), size(gt_image,2)]);
gt(gt_image(:, :, 1)==255 & gt_image(:, :, 2)==0 & gt_image(:, :, 3)==0)=1;
gt(gt==0)=2;

fore_x = 181:280;
fore_y = 221:320;
back_x = 116:215;
back_y = 1:100;

background_area = image(back_y, back_x, :);
foreground_area = image(fore_y, fore_x, :);

% Set number of components in mixture
k = 1;

% Get GMM for background
sz = size(background_area);
new_sz = sz(1) * sz(2);
background_area = double(reshape(background_area, [new_sz, 3]));
back_gmm = fitgmdist(background_area,k);

% Get GMM for foreground
sz = size(foreground_area);
new_sz = sz(1) * sz(2);
foreground_area = double(reshape(foreground_area, [new_sz, 3]));
fore_gmm = fitgmdist(foreground_area,k);

% precalculate probabilities
sz = size(image);
new_sz = sz(1) * sz(2);
reshaped_image = double(reshape(image, [new_sz, 3]));
sz = [ sz(1), sz(2) ];
back_prob = reshape(back_gmm.pdf(reshaped_image), sz);
fore_prob = reshape(fore_gmm.pdf(reshaped_image), sz);
probs = [back_prob(:)'; fore_prob(:)'];
%segment_init = MLE(ones([sz(1) * sz(2), 1]), probs);
%segment_init = reshape(segment_init, [sz(1), sz(2)]);
segment_init = randi(k, [sz(1), sz(2)]);
logprobs = -log(probs);

% perform segmentation
[map1, ~] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 1, 2, 10, 4);
[map2, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 1, 2, 10, 4);
[map3] = ICM(segment_init, logprobs, 2, 1, 50, 4);
[map4] = SimulatedAnnealing(segment_init, logprobs, 2, 1, 4, 0.6, 50, 4);
map_kmeans = kmeans(reshaped_image, 2);

dsc_swap = SimilarityScore(gt, map1, 2);
dsc_expansion = SimilarityScore(gt, map2, 2);
dsc_icm = SimilarityScore(gt, map3, 2);
dsc_sa = SimilarityScore(gt, map4, 2);
dsc_kmeans = SimilarityScore(gt, map_kmeans, 2);

[tpr_swap, tnr_swap, ~] = TruePositiveNegativeRates(gt, map1, 2);
[tpr_exp, tnr_exp, ~] = TruePositiveNegativeRates(gt, map2, 2);
[tpr_icm, tnr_icm, ~] = TruePositiveNegativeRates(gt, map3, 2);
[tpr_sa, tnr_sa, ~] = TruePositiveNegativeRates(gt, map4, 2);
[tpr_kmeans, tnr_kmeans, ~] = TruePositiveNegativeRates(gt, map_kmeans, 2);

map_kmeans = reshape(map_kmeans, size(map1));

%imagesc([map1, map3]);

mask = map1 == 2;
image_no_back = bsxfun(@times, image, cast(mask,class(image)));
imshow(image_no_back);
%imwrite( ind2rgb(im2uint8(mat2gray(map1)), parula(256)), 'data/bird.png')


%img = uint8(zeros(size(image)));
%for i=1:size(map1, 1)
%    for j=1:size(map1, 2)
%        if map1(i, j) == 2
%            img(i, j, :) = [ 255, 0, 0 ];
%        else
%            img(i, j, :) = uint8(image(i, j, :));
%        end
%    end
%end

%imwrite( img, 'data/bird.png');
