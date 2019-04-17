image = imread('data/2_horses_cropped.png');
imshow(image, []);
uiwait(msgbox('Draw on bird'));
hFH = imfreehand('Closed',false);
mask = createMask(hFH);
delete(hFH);

color_by_row = reshape(image, [size(image,1)*size(image,2), 3]);
fore = color_by_row(mask==1, :);

imshow(image, []);
uiwait(msgbox('Draw on background'));
hFH = imfreehand('Closed',false);
mask = createMask(hFH);
delete(hFH);

back = color_by_row(mask==1, :);

imshow(image, []);
uiwait(msgbox('Draw on ground'));
hFH = imfreehand('Closed',false);
mask = createMask(hFH);
delete(hFH);

ground = color_by_row(mask==1, :);

% show predefined ares
% imshow([background_area, foreground_area]);

% Set number of components in mixture
k = 4;

% Get GMM for background
back_gmm = FitGMMWithUnsetNumberOfComponents(double(back),k);

% Get GMM for foreground
fore_gmm = FitGMMWithUnsetNumberOfComponents(double(fore),k);

ground_gmm = FitGMMWithUnsetNumberOfComponents(double(ground),k);

% Precalculate probabilities
sz = size(image);
new_sz = sz(1) * sz(2);
reshaped_image = double(reshape(image, [new_sz, 3]));
sz = [ sz(1), sz(2) ];
back_prob = reshape(back_gmm.pdf(reshaped_image), sz);
fore_prob = reshape(fore_gmm.pdf(reshaped_image), sz);
ground_prob = reshape(ground_gmm.pdf(reshaped_image), sz);
probs = [back_prob(:)'; fore_prob(:)'; ground_prob(:)'];
segment_init = MLE(ones([sz(1) * sz(2), 1]), probs);
segment_init = reshape(segment_init, [sz(1), sz(2)]);
logprobs = -log(probs);
[map1, ~] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 1.5, 3, 5, 16);
[map2, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 1.5, 3, 5, 16);
imagesc([map1, map2]);

