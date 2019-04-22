image = imread('data/2_horses_cropped.png');
gt_image = imread('data/2_horses_cropped_gt.png');
gt = zeros([size(gt_image,1), size(gt_image,2)]);
gt(gt_image(:, :, 1)==255 & gt_image(:, :, 2)==0 & gt_image(:, :, 3)==0)=1;
gt(gt_image(:, :, 3)==255 & gt_image(:, :, 1)==0 & gt_image(:, :, 2)==0)=2;
gt(gt==0)=3;

obj1_x = 180:230;
obj1_y = 90:140;
obj2_x = 55:105;
obj2_y = 70:120;
back_x = 130:180;
back_y = 205:255;

obj1_area = image(obj1_y, obj1_x, :);
obj2_area = image(obj2_y, obj2_x, :);
back_area = image(back_y, back_x, :);

% Set number of components in mixture
k = 1;

% Get GMM for background
sz = size(back_area);
new_sz = sz(1) * sz(2);
back_area = double(reshape(back_area, [new_sz, 3]));
back_gmm = fitgmdist(back_area,k);

% Get GMM for foreground
sz = size(obj1_area);
new_sz = sz(1) * sz(2);
obj1_area = double(reshape(obj1_area, [new_sz, 3]));
obj1_gmm = fitgmdist(obj1_area,k);
sz = size(obj2_area);
new_sz = sz(1) * sz(2);
obj2_area = double(reshape(obj2_area, [new_sz, 3]));
obj2_gmm = fitgmdist(obj2_area,k);

% precalculate probabilities
sz = size(image);
new_sz = sz(1) * sz(2);
reshaped_image = double(reshape(image, [new_sz, 3]));
sz = [ sz(1), sz(2) ];
back_prob = reshape(back_gmm.pdf(reshaped_image), sz);
obj1_prob = reshape(obj1_gmm.pdf(reshaped_image), sz);
obj2_prob = reshape(obj2_gmm.pdf(reshaped_image), sz);
probs = [obj1_prob(:)'; obj2_prob(:)'; back_prob(:)'];
segment_init = MLE(ones([sz(1) * sz(2), 1]), probs);
segment_init = reshape(segment_init, [sz(1), sz(2)]);
logprobs = -log(probs);

% perform segmentation
[map1, ~] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 2, 3, 10, 8);
[map2, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 2, 3, 10, 8);
[map3] = ICM(segment_init, logprobs, 3, 2, 50, 8);
[map4] = SimulatedAnnealing(segment_init, logprobs, 3, 2, 4, 0.99, 50, 8);

dsc_swap = SimilarityScore(gt, map1, 2);
dsc_expansion = SimilarityScore(gt, map2, 2);
dsc_icm = SimilarityScore(gt, map3, 2);
dsc_sa = SimilarityScore(gt, map4, 2);

imagesc([map1, map2, map3, map4]);

mask = map2 == 3;
image_no_back = bsxfun(@times, image, cast(mask,class(image)));
imshow(image_no_back);

