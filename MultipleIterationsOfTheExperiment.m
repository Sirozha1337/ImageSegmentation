% Set parameters for random generation
Size = 300;
p = 10;
k = 3;
metricBasePath = 'experiment_data/metrics/';
segmentationBasePath = 'experiment_data/segmentations/';

%{
T_simple = cell2table(cell(0,8), 'VariableNames', ...
{
    'HMRF_MCEM_LastSample', 'HMRF_MCEM_MAP', ...
    'K_Means', 'GrabCut', ...
    'HMRF_EM_Expansion', 'HMRF_EM_Swap', ...
    'HMRF_VEM_LastSample', 'HMRF_VEM_MAP'
});
T_jaccard = cell2table(cell(0,8), 'VariableNames', ...
{
    'HMRF_MCEM_LastSample', 'HMRF_MCEM_MAP', ...
    'K_Means', 'GrabCut', ...
    'HMRF_EM_Expansion', 'HMRF_EM_Swap', ...
    'HMRF_VEM_LastSample', 'HMRF_VEM_MAP'
});
T_ssim = cell2table(cell(0,8), 'VariableNames', ...
{
    'HMRF_MCEM_LastSample', 'HMRF_MCEM_MAP', ...
    'K_Means', 'GrabCut', ...
    'HMRF_EM_Expansion', 'HMRF_EM_Swap', ...
    'HMRF_VEM_LastSample', 'HMRF_VEM_MAP'
});
%}
%results = zeros([100, 11]);
%T = readtable('exp_data.csv');
%start = 0;
%for i=1:100
%    if results(i, 1) == 0
%        start = i;
%        break;
%    end
%end

% Use pre-generated params for reproduceability 
load('good_params')
%start=0;
tic;
for i=81:100
    InitY = randi(k, [Size, Size]);
    [data, gt]=GenerateSynteticData(InitY, p, k, 2, mus, kappas, 20, 4);
    s.name = 'GroundTruth';
    s.data = reshape(data, [Size,Size,p]);
    s.segmentation = reshape(gt, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas;
    s.mus = mus;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');

    % MCEM
    [final_segm_mcem, ~, mus_mcem, kappas_mcem, final_segm_mcem2] = HMRF_MCEM(data, randi(k,[Size, Size]), k, 2, close_mus, close_kappas, 10, 10, 10, 4);
    
    s.name = 'HMRF_MCEM_LastSample';
    s.segmentation = reshape(final_segm_mcem, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_mcem;
    s.mus = mus_mcem;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
    
    s.name = 'HMRF_MCEM_MAP';
    s.segmentation = reshape(final_segm_mcem2, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_mcem;
    s.mus = mus_mcem;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
    
    % Kmeans
    final_segm_kmeans = kmeans(data, k);
    D = vmffactory(p);
    kappas_kmeans = zeros([1, k]);
    mus_kmeans = zeros([k, p]);
    for j=1:k
        data_in_k = data(final_segm_kmeans==j, :);
        params = D.estimatedefault(data_in_k');
        mus_kmeans(j, :) = squeeze(params.mu);
        kappas_kmeans(j) = params.kappa;
    end
    s.name = 'K_Means';
    s.segmentation = reshape(final_segm_kmeans, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_kmeans;
    s.mus = mus_kmeans;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
    

    % Grab Cut
    [final_segm_grc, mus_grc, kappas_grc] = Grab_Cut(data, reshape(final_segm_kmeans, [Size,Size]), k, 10, 40, close_mus, close_kappas, 5, 10, 4);
    s.name = 'Grab_Cut';
    s.segmentation = reshape(final_segm_grc, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_grc;
    s.mus = mus_grc;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
    
    % Hmrf EM
    [final_segm_expansion, ~, mus_expansion, kappas_expansion] = HMRF_EM(data, randi(k,[Size, Size]), k, 2, close_mus, close_kappas, 10, 10, 4, 'expansion');
    s.name = 'HMRF_EM_Expansion';
    s.segmentation = reshape(final_segm_expansion, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_expansion;
    s.mus = mus_expansion;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
    
    [final_segm_swap, ~, mus_swap, kappas_swap] = HMRF_EM(data, randi(k,[Size, Size]), k, 2, close_mus, close_kappas, 10, 10, 4, 'swap');
    s.name = 'HMRF_EM_Swap';
    s.segmentation = reshape(final_segm_expansion, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_swap;
    s.mus = mus_swap;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
    
    [final_segm_vem, ~, mus_vem, kappas_vem, final_segm_vem2] = HMRF_VEM(reshape(data, [Size,Size,p]), p, k, 2, close_mus, close_kappas, 0.5, 50, 20, 4);
    s.name = 'HMRF_VEM_LastSample';
    s.segmentation = reshape(final_segm_vem, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_swap;
    s.mus = mus_swap;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
    
    s.name = 'HMRF_VEM_MAP';
    s.segmentation = reshape(final_segm_vem2, [Size,Size]);
    s.beta = 2;
    s.kappas = kappas_swap;
    s.mus = mus_swap;
    save(strcat(segmentationBasePath, s.name, '_', num2str(i)),'s');
end
toc;
%boxplot([s_mcem, s_expansion, s_grc, s_swap, s_kmeans], 'Notch','on', 'Labels',{'mcem','hmrf-em (exp)', 'grab cut', 'hmrf-em (swap)', 'kmeans'});
