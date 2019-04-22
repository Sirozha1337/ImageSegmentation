%Модель наблюдения GMM, реальные rgb или grayscale изображения. Параметры оцениваются по выделенным пользователем областям. Считается MAP оценка.

image = imread('data/2_horses_cropped.png');
gt_image = imread('data/2_horses_cropped_gt.png');
gt = zeros([size(gt_image,1), size(gt_image,2)]);
gt(gt_image(:, :, 1)==255)=1;
gt(gt_image(:, :, 3)==255)=2;
gt(gt==0)=3;
imshow(image, []);
uiwait(msgbox('Draw on brown horse'));
hFH = imfreehand('Closed',false);
mask = createMask(hFH);
delete(hFH);

color_by_row = reshape(image, [size(image,1)*size(image,2), 3]);
obj1 = color_by_row(mask==1, :);

imshow(image, []);
uiwait(msgbox('Draw on white horse'));
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

% Set number of components in mixture
k = 1;

% Get GMM for object 1
obj1_gmm = FitGMMWithUnsetNumberOfComponents(double(obj1),k);

% Get GMM for object 2
obj2_gmm = FitGMMWithUnsetNumberOfComponents(double(obj2),k);

background_gmm = FitGMMWithUnsetNumberOfComponents(double(background),k);

% Precalculate probabilities
sz = size(image);
new_sz = sz(1) * sz(2);
reshaped_image = double(reshape(image, [new_sz, 3]));
sz = [ sz(1), sz(2) ];
obj1_prob = reshape(obj1_gmm.pdf(reshaped_image), sz);
obj2_prob = reshape(obj2_gmm.pdf(reshaped_image), sz);
ground_prob = reshape(background_gmm.pdf(reshaped_image), sz);
probs = [obj1_prob(:)'; obj2_prob(:)'; ground_prob(:)'];
mle = MLE(ones(sz), probs);
logprobs = -log(probs);
segment_init = randi(3, [sz(1), sz(2)]);
[map, ~] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 1, 3, 5, 16);

dsc = SimilarityScore(gt, map, 3);
