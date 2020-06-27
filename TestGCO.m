
%image = imread('data\2_horses_cropped.png');
image = NormalizeToUnitLength(double(imread('xray/data.png')));

width = size(image,1);
height = size(image,2);
p = size(image, 3);
data = reshape(double(image), [width*height, p]);
k = 10;
neighbours_count = 4;
beta = 2;

segment_init = kmeans(data, k, 'MaxIter', 500);
probs = zeros([k, width*height]);
logprobs = zeros([k, width*height]);
for i=1:k
    % достаем данные соответствующие размеченной области
    roi = data(segment_init==i, :);
    gmm = fitgmdist(roi,1);
    probs(i, :) = gmm.pdf(data)';
    logprobs(i, :) = -log(probs(i, :));
end
segment_init = reshape(segment_init, [width,height]);
[~, segment_init2] = max(probs, [], 1);
segment_init2 = reshape(segment_init2, [width,height]);

elements = numel(segment_init);
is = zeros(elements * 4, 1);
js = zeros(elements * 4, 1); 
vs = zeros(elements * 4, 1);
count = 1;
for i=1:elements
    if i - 1 >= 1
        is(count) = i;
        js(count) = i - 1;
        vs(count) = 1;
        count = count + 1;
    end
    if i + 1 <= elements
        is(count) = i;
        js(count) = i + 1;
        vs(count) = 1;
        count = count + 1;
    end
    if i - width >= 1
        is(count) = i;
        js(count) = i - width;
        vs(count) = 1;
        count = count + 1;
    end
    if i + width <= elements
        is(count) = i;
        js(count) = i + width;
        vs(count) = 1;
        count = count + 1;
    end
end
is = is(is ~= 0);
js = js(js ~= 0);
vs = vs(vs ~= 0);

neighbours = sparse(is, js, vs);
logprobs(logprobs==Inf) = 10^6;

h = GCO_Create(elements,k);
GCO_SetDataCost(h, logprobs);
GCO_SetNeighbors(h, neighbours);
tic;
GCO_Expansion(h, 150);
toc;
sample1 = GCO_GetLabeling(h);
sample1 = reshape(sample1, [width,height]);
GCO_Delete(h);

h = GCO_Create(elements,k);
GCO_SetDataCost(h, logprobs);
GCO_SetNeighbors(h, neighbours);
labelCosts = (1:k) * mean(logprobs(:)) * 100;
GCO_SetLabelCost(h, labelCosts);
tic;
GCO_Expansion(h, 150);
toc;
sample2 = GCO_GetLabeling(h);
sample2 = reshape(sample2, [width,height]);
GCO_Delete(h);


counts1 = zeros(1, k);
counts2 = zeros(1, k);
for i=1:k
    counts1(i) = sum(sample1(:)==i);
    counts2(i) = sum(sample2(:)==i);
end

figure('Name', 'Segment init')
imagesc(segment_init);
figure('Name', 'No Label Cost');
imagesc(sample1);
figure('Name', 'Label Cost');
imagesc(sample2);

%SaveImage(segment_init, 'work/horses_init.png');
%SaveImage(sample1, 'work/horses_no_cost.png');
%SaveImage(sample2, 'work/horses_with_cost.png');

