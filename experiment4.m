% ������ vMF, ������ �������������. ��������� HMRF-MCEM.

% ������ ���������
Size = 300;
p = 3;
k = 3;
kappas = repmat(10, 1, k);
mus = [[-0.57, 0.57, 0.57];[0, -1, 0]; [1, 0, 0];];
close_mus = [[-0.21, 0.64, 0.13];[0.5, -1, 0]; [1, 0.05, 0.45];];
close_kappas = [5, 8, 7];

% ���������� ��������� ������� � ������� �� 1 �� k
InitY = randi(k, [Size, Size]);
[data, gt]=GenerateSynteticData(InitY, p, k, 2, mus, kappas, 20, 4);

% ����� �� ��������� ������������
[final_segm_rand, beta_rand, mus_rand, kappas_rand] = HMRF_MCEM(data, [Size, Size], k, 2, close_mus, close_kappas, 10, 10, 5, 4);

% ����� � MLE
[probs, ~] = CalculateLikelihoodProbabilities(data, k, close_kappas, close_mus);
mle = MLE(zeros([Size * Size, 1]), probs);
mle = reshape(mle, [Size, Size]);
[final_segm_mle, beta_mle, mus_mle, kappas_mle] = HMRF_MCEM(data, [Size, Size], k, 2, close_mus, close_kappas, 10, 10, 5, 4, mle);

% ����� � kmeans
segment_init_kmeans = kmeans(data, k, 'MaxIter',500);
vmf = vmffactory(p);
mus_kmeans = zeros(k, p);
kappas_kmeans = zeros(1, k);
for i=1:k
    % ������� ������ ��������������� ����������� �������
    roi = data(segment_init_kmeans==i, :);
    [theta] = vmf.estimatedefault(roi');
    mus_kmeans(i, :) = squeeze(theta.mu);
    kappas_kmeans(i) = theta.kappa;
end
[final_segm_kmeans, beta_kmeans, mus_kmeans, kappas_kmeans] = HMRF_MCEM(data, [Size, Size], k, 2, mus_kmeans, kappas_kmeans, 10, 10, 5, 4);

dsc_rand = SimilarityScore(gt, final_segm_rand, k);
dsc_mle = SimilarityScore(gt, final_segm_mle, k);
dsc_kmeans = SimilarityScore(gt, final_segm_kmeans, k);

