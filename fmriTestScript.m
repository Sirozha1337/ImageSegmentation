% читаем данные
data = niftiread('D:\Study\Master\Preprocessing\sub38579\prepros\fmri.nii.gz');
anat = niftiread('D:\Study\Master\Preprocessing\sub38579\prepros\anat.nii.gz');
anat2 = RotateMatrix(anat, 2);
gray = niftiread('D:\Study\Master\Preprocessing\sub38579\prepros\gray_matter.nii.gz');

% this command can be used to display different slices of an anatomical and
% functional image at once
% spm_check_registration('D:\Study\Master\Preprocessing\sub38579\prepros2\anat.nii.gz', ['D:\Study\Master\Preprocessing\sub38579\prepros2\fmri.nii.gz' ',1'], 'D:\Study\Master\Preprocessing\sub38579\prepros2\gray_matter.nii.gz')

[data2, prepadding, postpadding] = RealignDataByMask(data, gray);
[gray2, prepadding_mask, postpadding_mask] = RealignDataByMask(gray, gray);
sz = size(data2);
data2(gray2 == 0) = 0;

% раскладываем данные по рядам и нормализуем
k = 3;
timeseries_by_rows = reshape(data2, [prod(sz(1:3)), sz(4)]);
timeseries_by_rows = double(timeseries_by_rows); 
lenghts = sqrt(sum(timeseries_by_rows .^ 2, 2));
non_empty_rows = lenghts > 0;
non_empty_rows_indexes = find(lenghts > 0);
timeseries_by_rows = timeseries_by_rows - mean(timeseries_by_rows(:));
timeseries_by_rows = NormalizeToUnitLength(timeseries_by_rows);

% получаем начальную сегментацию
segment_init_kmeans = kmeans(timeseries_by_rows(gray2(:)~=0), k, 'MaxIter',500);
segment_init = zeros([size(timeseries_by_rows, 1), 1]);
segment_init(gray2(:)~=0) = segment_init_kmeans;

% инициализируем структуры для параметров распределения
vmf = vmffactory(197);
mus = zeros(k, 197);
kappas = zeros(1, k);
thetas = cell(k, 1);
for i=1:k
    % достаем данные соответствующие размеченной области
    roi = timeseries_by_rows(gray2(:)~=0 & segment_init==i, :);
    [theta] = vmf.estimatedefault(roi');
    thetas{i} = theta;
    mus(i, :) = squeeze(theta.mu);
    kappas(i) = theta.kappa;
end

% calculate MLE
segm_vmf = zeros([size(timeseries_by_rows, 1), 1]);
prob_vmf = zeros([size(timeseries_by_rows, 1), 1]);
for i=1:k
    theta = thetas{i};
    tmp = vmf.pdf(theta, timeseries_by_rows')';
    segm_vmf(prob_vmf < tmp) = i;
    prob_vmf = max(tmp, prob_vmf);
end
segment_init = reshape(segment_init, sz(1:3));
segment_init2 = PadData(segment_init, prepadding, postpadding);
figure();
ShowImageWithLabels(anat, segment_init2, 3, 25);

segm_vmf = reshape(segm_vmf, sz(1:3));
segm_vmf2 = PadData(segm_vmf, prepadding, postpadding);
figure();
ShowImageWithLabels(anat, segm_vmf2, 3, 25);
% HMRF-MCEM
[final_segm_mcem, beta_mcem, mus_mcem, kappas_mcem, final_segm_mcem_map] = HMRF_MCEM(timeseries_by_rows, segment_init, k, 2, mus, kappas, 10, 10, 10, 6, gray2);
final_segm_mcem2 = PadData(final_segm_mcem, prepadding, postpadding);
figure();
ShowImageWithLabels(anat, final_segm_mcem2, 3, 25);

% HMRF-VEM
[final_segm_vem, beta_vem, mus_vem, kappas_vem, final_segm_vem_map] = HMRF_VEM(reshape(timeseries_by_rows, sz), 197, k, 2, mus, kappas, 0.5, 50, 25, 6, gray2);
final_segm_vem2 = PadData(final_segm_vem_map, prepadding, postpadding);
figure();
ShowImageWithLabels(anat, final_segm_vem2, 3, 25);

% Grab Cut

[final_segm_grc, mus_grc, kappas_grc, ...
    all_mus_grc, all_kappas_grc] = Grab_Cut(timeseries_by_rows, reshape(segment_init, sz(1:3)), k, 10, 40, mus, kappas, 5, 10, 6);
final_segm_grc2 = PadData(final_segm_grc, prepadding, postpadding);
figure();
ShowImageWithLabels(anat, final_segm_grc2, 3, 25);

% HMRF EM (Alpha Expansion)
[final_segm_expansion, beta_expansion, mus_expansion, kappas_expansion, ...
    all_mus_expansion, all_kappas_expansion] = HMRF_EM(timeseries_by_rows, reshape(segment_init, sz(1:3)), k, 2, mus, kappas, 10, 10, 6, 'expansion');
final_segm_expansion2 = PadData(final_segm_expansion, prepadding, postpadding);
figure();
ShowImageWithLabels(anat, final_segm_expansion2, 3, 25);

% HMRF EM (Alpha-Beta Swap)
% [final_segm_swap, beta_swap, mus_swap, kappas_swap, ...
%    all_mus_swap, all_kappas_swap] = HMRF_EM(timeseries_by_rows, reshape(segment_init, sz(1:3)), k, 2, mus, kappas, 10, 10, 6, 'swap');


gt = final_segm_mcem_map;
s_dsc_mcem = SimpleSimilarityScore(gt, final_segm_mcem, k);
[s_dsc_kmeans, s_mkm] = SimpleSimilarityScore(gt, segment_init, k);
[s_dsc_vem, ~] = SimpleSimilarityScore(gt, final_segm_vem_map, k);
[s_dsc_grc, ~] = SimpleSimilarityScore(gt, final_segm_grc, k);
[s_dsc_expansion, s_mem] = SimpleSimilarityScore(gt, final_segm_expansion, k);
% s_dsc_swap = SimpleSimilarityScore(gt, final_segm_swap, k);

[s_kmeans, j_kmeans, ssim_kmeans] = CalculateScores(gt, segment_init, k);
[s_vem, j_vem, ssim_vem] = CalculateScores(gt, final_segm_vem_map, k);
[s_grc, j_grc, ssim_grc] = CalculateScores(gt, final_segm_grc, k);
[s_em, j_em, ssim_em] = CalculateScores(gt, final_segm_expansion, k);

s_grc2 = PadData(reshape(s_grc, size(final_segm_vem)), prepadding, postpadding);
s_grc2 = RotateMatrix(s_grc2, 2);
f = ShowMultipleSlicesWithLabels(anat2, s_grc2, [1,2,3], [45,55,45], ["Transverse", "Frontal", "Saggital"]);
saveas(f, 'fmri/GrabCut.png');

s_vem2 = PadData(reshape(s_vem, size(final_segm_vem)), prepadding, postpadding);
s_vem2 = RotateMatrix(s_vem2, 2);
f = ShowMultipleSlicesWithLabels(anat2, s_vem2, [1,2,3], [45,55,45], ["Transverse", "Frontal", "Saggital"]);


s_vem_map2 = PadData(reshape(s_vem, size(final_segm_vem_map)), prepadding, postpadding);
s_vem_map2 = RotateMatrix(s_vem_map2, 2);
f = ShowMultipleSlicesWithLabels(anat2, s_vem_map2, [1,2,3], [45,55,45], ["Transverse", "Frontal", "Saggital"]);
saveas(f, 'fmri/HMRF-VEM.png')

final_segm_mcem2 = PadData(reshape(final_segm_mcem, size(final_segm_vem)), prepadding, postpadding);
final_segm_mcem2 = RotateMatrix(final_segm_mcem2, 2);
f = ShowMultipleSlicesWithLabels(anat2, final_segm_mcem2, [1,2,3], [45,55,45], ["Transverse", "Frontal", "Saggital"]);

final_segm_mcem_map2 = PadData(reshape(final_segm_mcem_map, size(final_segm_mcem_map)), prepadding, postpadding);
final_segm_mcem_map2 = RotateMatrix(final_segm_mcem_map2, 2);
f = ShowMultipleSlicesWithLabels(anat2, final_segm_mcem_map2, [1,2,3], [45,55,45], ["Transverse", "Frontal", "Saggital"]);
saveas(f, 'fmri/HMRF-MCEM.png')

s_mkm2 = PadData(reshape(s_mkm, size(final_segm_vem)), prepadding, postpadding);
s_mkm2 = RotateMatrix(s_mkm2, 2);
f = ShowMultipleSlicesWithLabels(anat2, s_mkm2, [1,2,3], [45,55,45], ["Transverse", "Frontal", "Saggital"]);
saveas(f, 'fmri/KMEANS.png')

s_mem2 = PadData(reshape(s_mem, size(final_segm_vem)), prepadding, postpadding);
s_mem2 = RotateMatrix(s_mem2, 2);
f = ShowMultipleSlicesWithLabels(anat2, s_mem2, [1,2,3], [45,55,45], ["Transverse", "Frontal", "Saggital"]);
saveas(f, 'fmri/HMRF-EM.png')


%{
Size = [64,64,39];
tmp = reshape(gt, Size);
SaveImage(tmp(:,:,10), 'data/fmri_gt2.png');
tmp = reshape(final_segm_grc, Size);
SaveImage(tmp(:,:,10), 'data/fmri_grc2.png');
tmp = reshape(final_segm_expansion, Size);
SaveImage(tmp(:,:,10), 'data/fmri_expansion2.png');
tmp = reshape(final_segm_swap, Size);
SaveImage(tmp(:,:,10), 'data/fmri_swap2.png');
final_segm_kmeans = segment_init;
tmp = reshape(final_segm_kmeans, Size);
SaveImage(tmp(:,:,10), 'data/fmri_kmeans2.png');
tmp = mean(data, 4);
SaveImage(tmp(:,:,10), 'data/fmri_data2.png');
%}
