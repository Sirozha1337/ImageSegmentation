% ������ ������
data = Read4DArrayFromStarplus(4847, 5);
mdata = mean(data,4);
imagesc(mdata(:,:,1));
roi_names = ["CALC", "LIPL_LT_LDLPFC", "LTRIA", "LOPER", "LIPS" ];
%roi_names = ["CALC_LFEF_RFEF_LSGA_RSGA_LSPL_RSPL_LIT_RIT", "LDLPFC_RDLPFC_LOPER_ROPER", "LIFG_LPPREC_RPPREC_LTRIA_RTRIA_LIPL_RIPL", "LT_RT", "SMA_LIPS_RIPS"];
%roi_names = ["CALC_LFEF_RFEF_LSGA_RSGA_LSPL_RSPL_LIT_RIT", "LDLPFC_RDLPFC_LOPER_ROPER"];
k = 6;
% ������������ ������ �� ����� � �����������
timeseries_by_rows = reshape(data, [size(data, 1) * size(data, 2) * size(data, 3), size(data, 4)]);
timeseries_by_rows = double(timeseries_by_rows);
% ������� �������� �������, �����, ����� ������� �� ������������ ������
non_empty_rows = sqrt(sum(timeseries_by_rows .^ 2, 2)) ~= 0;
%timeseries_by_rows = timeseries_by_rows - mean(timeseries_by_rows(:));
timeseries_by_rows = NormalizeToUnitLength(timeseries_by_rows);

% �������������� ��������� ��� ���������� �������������
vmf = vmffactory(size(data, 4));
mus = zeros(k, size(data, 4));
kappas = zeros(1, k);
% ��������������� ����� ���� ROI, �����, ����� ����� �� ROI �������
sum_roi_mask = zeros([size(timeseries_by_rows,1), 1]);
% ��������� ��������� �� ROI
groundTruth = zeros([size(data, 1), size(data, 2), size(data, 3)]);
for i=1:(k-1)
    roi_name = roi_names(i);
    % ������� ������ ��������������� ����������� �������
    roi_mask = ReadROIFromStarplus(4847, roi_name);
    groundTruth = groundTruth + roi_mask * i;
    roi_mask = roi_mask(:);
    sum_roi_mask = sum_roi_mask | roi_mask;
    roi = timeseries_by_rows(non_empty_rows & roi_mask==1, :);
    roi = roi(sqrt(sum(roi .^ 2, 2)) ~= 0, :);
    [theta] = vmf.estimatedefault(roi');
    mus(i, :) = squeeze(theta.mu);
    kappas(i) = theta.kappa;
end
%

% ������������� �� ROI �������
non_roi_mask = ~sum_roi_mask;
groundTruth = groundTruth + reshape(k .* non_roi_mask .* (sqrt(sum(timeseries_by_rows .^ 2, 2)) ~= 0), size(groundTruth));
roi = timeseries_by_rows(non_roi_mask==1, :);
roi = roi(sqrt(sum(roi .^ 2, 2)) ~= 0, :);
[theta] = vmf.estimatedefault(roi');
mus(k, :) = squeeze(theta.mu);
kappas(k) = theta.kappa;
% kappas = ones(1,3);
% ��������� �������� mu
% dist = squareform(pdist(mus, 'cosine'));

% ������� ����������� P � -logP
[probs, logprobs] = CalculateLikelihoodProbabilities(timeseries_by_rows, k, kappas, mus);
% ������� ������ ������������� �������������
segment_init = MLE(timeseries_by_rows, probs);
% ������ �������� ����������� ����� 0
segment_init(~non_empty_rows) = 0;
% �������������� ����������� � ���������� �������
segment_init = reshape(segment_init, [size(data,1), size(data,2), size(data,3)]);
% ���� MAP-������
[map, energy] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 1, k, 10);
[beta, mu, kappa] = EstimateParametersOneSample(reshape(timeseries_by_rows, size(data)), map, energy, k, 13, 1, mus, kappas);
mdata = mean(data, 4);
z = 3;
imagesc([segment_init(:,:,z),map(:,:,z),groundTruth(:,:,z)]);
figure();
imagesc(mdata(:,:,z));