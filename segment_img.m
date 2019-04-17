
image = imread('D:\Study\Master\implementations\data\a_8049.jpg');
% Define background and foreground
back_x1 = 1; back_x2 = 100; back_y1 = 116; back_y2 = 215;
fore_x1 = 221; fore_x2 = 320; fore_y1 = 181; fore_y2 = 280;

background_area = image(back_x1:back_x2, back_y1:back_y2, :);
foreground_area = image(fore_x1:fore_x2, fore_y1:fore_y2, :);

% show predefined ares
% imshow([background_area, foreground_area]);

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

% Precalculate probabilities
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
[map1, ~] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 2, 2, 10, 8);
[map2, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 2, 2, 10, 8);
imagesc([map1, map2]);
