image = imread('data/2_horses_cropped.png');
imshow(image, []);
uiwait(msgbox('Draw on one horse'));
hFH = imfreehand('Closed',false);
mask = createMask(hFH);
delete(hFH);

color_by_row = reshape(image, [size(image,1)*size(image,2), 3]);
obj1 = color_by_row(mask==1, :);

imshow(image, []);
uiwait(msgbox('Draw on second horse'));
hFH = imfreehand('Closed',false);
mask = createMask(hFH);
delete(hFH);

obj2 = color_by_row(mask==1, :);

imshow(image, []);
uiwait(msgbox('Draw on background'));
hFH = imfreehand('Closed',false);
mask = createMask(hFH);
delete(hFH);

background = color_by_row(mask==1, :);

% show predefined ares
% imshow([background_area, foreground_area]);

% Set number of components in mixture
k = 4;

% Get GMM for background
obj2_gmm = FitGMMWithUnsetNumberOfComponents(double(obj2),k);

% Get GMM for foreground
obj1_gmm = FitGMMWithUnsetNumberOfComponents(double(obj1),k);

background_gmm = FitGMMWithUnsetNumberOfComponents(double(background),k);

% Precalculate probabilities
sz = size(image);
new_sz = sz(1) * sz(2);
reshaped_image = double(reshape(image, [new_sz, 3]));
sz = [ sz(1), sz(2) ];
obj2_prob = reshape(obj2_gmm.pdf(reshaped_image), sz);
obj1_prob = reshape(obj1_gmm.pdf(reshaped_image), sz);
ground_prob = reshape(background_gmm.pdf(reshaped_image), sz);
probs = [obj2_prob(:)'; obj1_prob(:)'; ground_prob(:)'];
segment_init = MLE(ones([sz(1) * sz(2), 1]), probs);
segment_init = reshape(segment_init, [sz(1), sz(2)]);
logprobs = -log(probs);
[map1, ~] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 1.5, 3, 5, 16);
[map2, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 1.5, 3, 5, 16);
imagesc([map1, map2]);

