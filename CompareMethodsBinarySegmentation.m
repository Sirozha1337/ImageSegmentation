
image = imread('data/carriage.png');
gt_image = imread('data/carriage_gt.png');
gt = zeros([size(gt_image,1), size(gt_image,2)]);
gt(gt_image(:, :, 1)==255 & gt_image(:, :, 2)==0 & gt_image(:, :, 3)==0)=1;
gt(gt==0)=2;

fore_x = 70:120;
fore_y = 80:130;
back_x = 240:290;
back_y = 110:160;

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
segment_init = MLE(ones([sz(1) * sz(2), 1]), probs);
segment_init = reshape(segment_init, [sz(1), sz(2)]);
logprobs = -log(probs);

% perform segmentation
[map1, ~] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 2, 2, 10, 8);
[map2, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 2, 2, 10, 8);
[map3] = ICM(segment_init, logprobs, 2, 2, 50, 8);
[map4] = SimulatedAnnealing(segment_init, logprobs, 2, 2, 4, 0.99, 50, 8);

dsc_swap = SimilarityScore(gt, map1, 2);
dsc_expansion = SimilarityScore(gt, map2, 2);
dsc_icm = SimilarityScore(gt, map3, 2);
dsc_sa = SimilarityScore(gt, map4, 2);

imagesc([map1, map2, map3, map4]);

mask = map2 == 2;
image_no_back = bsxfun(@times, image, cast(mask,class(image)));
imshow(image_no_back);

