% Set parameters for random generation
Size = 300;
p = 10;
k = 3;

%D = vmffactory(p);
%kappas = repmat(10, 1, k);
%close_kappas = kappas + 2 * rand(size(kappas)) - 1;
%mus = zeros([k, p]);
%close_mus = zeros([k, p]);
%for i=1:k
%    mus(i, :) = D.sample(struct('mu', zeros([p, 1]), 'kappa', 0), 1)';
%    close_mu = mus(i, :) + rand(size(mus(i, :)));
%    close_mus(i, :) = close_mu / sqrt(sum(close_mu .^ 2));
%end
% Generate starting configuration
%InitY = randi(k, [Size, Size]);
%[data, gt]=GenerateSynteticData(InitY, p, k, 2, mus, kappas, 20, 4);
%imwrite(reshape(data, [Size, Size, 3]), 'data/generated.png');
%save('good_params','mus','kappas', 'close_mus', 'close_kappas', 'data', 'gt')

% Use pre-generated params for reproduceability 
load('good_params')

% MCEM
[final_segm_mcem, beta_mcem, mus_mcem, kappas_mcem, ...
    all_mus_mcem, all_kappas_mcem] = HMRF_MCEM(data, randi(k, [Size, Size]), k, 2, close_mus, close_kappas, 10, 10, 10, 4);
[score_mcem, final_segm_mcem] = SimilarityScore(gt, final_segm_mcem, k);
ssim_mcem = ssim(gt(:), final_segm_mcem(:));


% Kmeans
final_segm_kmeans = kmeans(data, k);
D = vmffactory(p);
kappas_kmeans = zeros([1, k]);
mus_kmeans = zeros([k, p]);
for i=1:k
    data_in_k = data(final_segm_kmeans==i, :);
    params = D.estimatedefault(data_in_k');
    mus_kmeans(i, :) = squeeze(params.mu);
    kappas_kmeans(i) = params.kappa;
end
[score_kmeans, final_segm_kmeans] = SimilarityScore(gt, final_segm_kmeans, k);
ssim_kmeans = ssim(gt(:), final_segm_kmeans(:));

% Mean Field
[final_segm_vem, beta_vem, mus_vem, kappas_vem, final_segm_vem2] = HMRF_VEM(reshape(data, [Size,Size,p]), p, k, 2, close_mus, close_kappas, 0.5, 50, 20, 4);
[score_vem, final_segm_vem] = SimilarityScore(gt, final_segm_vem, k);
ssim_vem = ssim(gt(:), final_segm_vem(:));

% Grab Cut
[final_segm_grc, mus_grc, kappas_grc, ...
    all_mus_grc, all_kappas_grc] = Grab_Cut(data, reshape(final_segm_kmeans, [Size,Size]), k, 10, 40, close_mus, close_kappas, 5, 10, 4);
[~, final_segm_grc] = SimilarityScore(gt, final_segm_grc, k);
ssim_grc = ssim(gt(:), final_segm_grc(:));

% Hmrf EM
[final_segm_expansion, beta_expansion, mus_expansion, kappas_expansion, ...
    all_mus_expansion, all_kappas_expansion] = HMRF_EM(data, randi(k, [Size, Size]), k, 2, close_mus, close_kappas, 10, 10, 4, 'expansion');
[~, final_segm_expansion] = SimilarityScore(gt, final_segm_expansion, k);
ssim_expansion = ssim(gt(:), final_segm_expansion(:));


[final_segm_swap, beta_swap, mus_swap, kappas_swap, ...
    all_mus_swap, all_kappas_swap] = HMRF_EM(data, randi(k, [Size, Size]), k, 2, close_mus, close_kappas, 10, 10, 4, 'swap');
[~, final_segm_swap] = SimilarityScore(gt, final_segm_swap, k);
ssim_swap = ssim(gt(:), final_segm_swap(:));

%imwrite( ind2rgb(im2uint8(mat2gray(final_segm_grc)), parula(256)), 'data/generated2.png')

% Calculate similarity scores
%{
dsc_mcem = SimilarityScore(gt, final_segm_mcem, k);
dsc_grc = SimilarityScore(gt, final_segm_grc, k);
[dsc_expansion, mem] = SimilarityScore(gt, final_segm_expansion, k);
dsc_swap = SimilarityScore(gt, final_segm_swap, k);
[dsc_kmeans, mkm] = SimilarityScore(gt, final_segm_kmeans, k);

s_dsc_mcem = SimpleSimilarityScore(gt, final_segm_mcem, k);
s_dsc_grc = SimpleSimilarityScore(gt, final_segm_grc, k);
[s_dsc_expansion, s_mem] = SimpleSimilarityScore(gt, final_segm_expansion, k);
s_dsc_swap = SimpleSimilarityScore(gt, final_segm_swap, k);
[s_dsc_kmeans, s_mkm] = SimpleSimilarityScore(gt, final_segm_kmeans, k);
%}
% Saving resulting images
%imagesc(reshape(gt, [Size,Size]));
%figure(); imagesc(reshape(final_segm_grc, [Size,Size]));
%figure(); imagesc(reshape(final_segm_mcem, [Size,Size]));
%figure(); imagesc(reshape(final_segm_kmeans, [Size,Size]));
%SaveImage(reshape(gt, [Size,Size]), 'data/sim_gt.png');
%SaveImage(reshape(mkm, [Size,Size]), 'data/sim_kmeans.png');
%SaveImage(reshape(mem, [Size,Size]), 'data/sim_em.png');
%SaveImage(reshape(final_segm_mcem, [Size,Size]), 'data/sim_mcem.png');
%SaveImage(reshape(s_grc, [Size,Size]), 'data/sim_grc.png');
%SaveImage(mean(reshape(data, [Size,Size, p]), 3), 'data/sim_data.png');

% Saving plots
%fig = PlotDistanceToTruth(mus, all_mus_mcem, 1, 'Distance to real mu', 'HMRF-MCEM');
%saveas(fig, 'data/dist_mu_mcem.png');
%fig = PlotDistanceToTruth(kappas, squeeze(all_kappas_mcem), 0, 'Distance to real kappa', 'HMRF-MCEM');
%saveas(fig, 'data/dist_kappa_mcem.png');
%fig = PlotDistanceToTruth(mus, all_mus_expansion, 1, 'Distance to real mu', 'HMRF-EM');
%saveas(fig, 'data/dist_mu_em.png');
%fig = PlotDistanceToTruth(kappas, squeeze(all_kappas_expansion), 0, 'Distance to real kappa', 'HMRF-EM');
%saveas(fig, 'data/dist_kappa_em.png');
%fig = PlotDistanceToTruth(mus, all_mus_grc, 1, 'Distance to real mu', 'GrabCut');
%saveas(fig, 'data/dist_mu_grc.png');
%fig = PlotDistanceToTruth(kappas, squeeze(all_kappas_grc), 0, 'Distance to real kappa', 'GrabCut');
%saveas(fig, 'data/dist_kappa_grc.png');



