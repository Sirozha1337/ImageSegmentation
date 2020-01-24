% Set parameters for random generation
Size = 300;
p = 10;
k = 3;
metricBasePath = 'experiment_data/metrics/';
segmentationBasePath = 'experiment_data/segmentations/';
starti=1;
endi=100;

tmp = zeros(endi, 1);
T_simple = table(tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp, 'VariableNames', ...
{
    'HMRF_MCEM_LastSample', 'HMRF_MCEM_MAP', ...
    'K_Means', 'Grab_Cut', ...
    'HMRF_EM_Expansion', 'HMRF_EM_Swap', ...
    'HMRF_VEM_LastSample', 'HMRF_VEM_MAP'
});
T_jaccard = table(tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp, 'VariableNames', ...
{
    'HMRF_MCEM_LastSample', 'HMRF_MCEM_MAP', ...
    'K_Means', 'Grab_Cut', ...
    'HMRF_EM_Expansion', 'HMRF_EM_Swap', ...
    'HMRF_VEM_LastSample', 'HMRF_VEM_MAP'
});
T_ssim = table(tmp, tmp, tmp, tmp, tmp, tmp, tmp, tmp, 'VariableNames', ...
{
    'HMRF_MCEM_LastSample', 'HMRF_MCEM_MAP', ...
    'K_Means', 'Grab_Cut', ...
    'HMRF_EM_Expansion', 'HMRF_EM_Swap', ...
    'HMRF_VEM_LastSample', 'HMRF_VEM_MAP'
});

algos = [
    "HMRF_MCEM_LastSample", "HMRF_MCEM_MAP", ...
    "K_Means", "Grab_Cut", ...
    "HMRF_EM_Expansion", "HMRF_EM_Swap", ...
    "HMRF_VEM_LastSample", "HMRF_VEM_MAP"
];

for i=starti:endi
    gt_struct = load(strcat(segmentationBasePath, 'GroundTruth', '_', num2str(i)),'s');
    gt = gt_struct.s.segmentation;
    for algo=algos
        res_struct = load(strcat(segmentationBasePath, algo, '_', num2str(i)),'s');
        [simple_score, best_segm] = SimpleSimilarityScore(gt, res_struct.s.segmentation, k);
        [jaccard_score, best_segm2] = SimilarityScore(gt, best_segm, k);
        sz = size(res_struct.s.segmentation);
        ssim_score1 = ssim(reshape(best_segm,sz), reshape(gt,sz));
        if ssim_score1 < 0.5
            imagesc(reshape(best_segm,sz));
            figure();  imagesc(reshape(gt,sz));
            a = 1;
        end
        ssim_score2 = ssim(reshape(best_segm2,sz), reshape(gt,sz));
        ssim_score = max([ssim_score1,ssim_score2]);
        T_simple(i, char(algo)) = { simple_score };
        T_jaccard(i, char(algo)) = { jaccard_score };
        T_ssim(i, char(algo)) = { ssim_score };
    end
end
    
writetable(T_simple, strcat(metricBasePath, 'simple.csv'));
writetable(T_jaccard, strcat(metricBasePath, 'jaccard.csv'));
writetable(T_ssim, strcat(metricBasePath, 'ssim.csv'));

T_simple = readtable(strcat(metricBasePath, 'simple.csv'));
T_jaccard = readtable(strcat(metricBasePath, 'jaccard.csv'));
T_ssim = readtable(strcat(metricBasePath, 'ssim.csv'));
figure;  
subplot(1,2,1)
%suptitle('Simple Similarity')
%{
boxplot([
    T_simple.HMRF_MCEM_LastSample, ...
    T_simple.HMRF_VEM_LastSample, ...
    T_simple.Grab_Cut, ...
    T_simple.K_Means], ...
    'Notch','on', 'Labels', ...
    {'HMRF-MCEM','HMRF-VEM', 'Grab-Cut', 'K-Means'});
subplot(1,3,2)
%}
%suptitle('Jaccard')
boxplot([
    T_jaccard.HMRF_MCEM_LastSample, ...
    T_jaccard.HMRF_VEM_LastSample, ...
    T_jaccard.Grab_Cut, ...
    T_jaccard.K_Means], ...
    'Notch','on', 'Labels', ...
    {'HMRF-MCEM','HMRF-VEM', 'GrabCut', 'K-Means'});
subplot(1,2,2)
%suptitle('SSIM')
boxplot([
    T_ssim.HMRF_MCEM_LastSample, ...
    T_ssim.HMRF_VEM_LastSample, ...
    T_ssim.Grab_Cut, ...
    T_ssim.K_Means], ...
    'Notch','on', 'Labels', ...
    {'HMRF-MCEM','HMRF-VEM', 'GrabCut', 'K-Means'});

currentFigure = gcf;
title(currentFigure.Children(1), 'SSIM');
title(currentFigure.Children(2), 'Jaccard');
%title(currentFigure.Children(3), 'Simple Similarity');