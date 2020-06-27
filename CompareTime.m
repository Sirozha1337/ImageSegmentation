Size = 300;
p = 10;
k = 3;
segmentationBasePath = 'experiment_data/segmentations/';
load('good_params');
%close_kappas = close_kappas - rand(1,3) * 5;

s = load(strcat(segmentationBasePath, 'GroundTruth', '_', num2str(1)));
data = reshape(s.s.data, [Size * Size,p]);
gt = s.s.segmentation;

iterations = [6,10,20,30,40,50];
iterations_len = 6;
timings = zeros(iterations_len,1);
simple_scores = zeros(iterations_len,1);
jaccard_scores = zeros(iterations_len,1);
ssim_scores = zeros(iterations_len,1);
simple_scores2 = zeros(iterations_len,1);
jaccard_scores2 = zeros(iterations_len,1);
ssim_scores2 = zeros(iterations_len,1);
for i=1:iterations_len
    iters = iterations(i);
    burn_in = iters / 2;
    num_samples = iters / 2;
    tic;
    [final_segm_mcem, ~, ~, ~, final_segm_mcem2] = HMRF_MCEM(data, randi(k,[Size, Size]), k, 2, close_mus, close_kappas, burn_in, num_samples, 10, 4);
    timings(i) = toc;
    
    [simple_scores(i), jaccard_scores(i), ssim_scores(i)] = CalculateScores(gt, final_segm_mcem, k);
    [simple_scores2(i), jaccard_scores2(i), ssim_scores2(i)] = CalculateScores(gt, final_segm_mcem2, k);
end


vem_timings = zeros(iterations_len,1);
vem_simple_scores = zeros(iterations_len,1);
vem_jaccard_scores = zeros(iterations_len,1);
vem_ssim_scores = zeros(iterations_len,1);
vem_simple_scores2 = zeros(iterations_len,1);
vem_jaccard_scores2 = zeros(iterations_len,1);
vem_ssim_scores2 = zeros(iterations_len,1);
for i=1:iterations_len
    iters = iterations(i);
    tic;
    [final_segm_vem, ~, ~, ~, final_segm_vem2] = HMRF_VEM(reshape(data, [Size,Size,p]), p, k, 2, close_mus, close_kappas, 0.5, 50, iters, 4);
    vem_timings(i) = toc;
    
    [vem_simple_scores(i), vem_jaccard_scores(i), vem_ssim_scores(i)] = CalculateScores(gt, final_segm_vem, k);
    [vem_simple_scores2(i), vem_jaccard_scores2(i), vem_ssim_scores2(i)] = CalculateScores(gt, final_segm_vem2, k);
end

figure();
plot(iterations, timings);
figure();
plot(iterations, vem_timings);
%{
boxplot([
    vem_simple_scores, ...
    vem_simple_scores2, ...
    simple_scores, ...
    simple_scores2], ...
    'Notch','on', 'Labels', ...
    {'HMRF-VEM1','HMRF-VEM2', 'HMRF-MCEM1', 'HMRF-MCEM2'});
%}
function[simple_score, jaccard_score, ssim_score] = CalculateScores(gt, segm, k)
    [simple_score, best_segm] = SimpleSimilarityScore(gt, segm, k);
    [jaccard_score, best_segm2] = SimilarityScore(gt, best_segm, k);
    sz = size(gt);
    ssim_score1 = ssim(reshape(best_segm,sz), reshape(gt,sz));
    ssim_score2 = ssim(reshape(best_segm2,sz), reshape(gt,sz));
    ssim_score = max([ssim_score1,ssim_score2]);
end