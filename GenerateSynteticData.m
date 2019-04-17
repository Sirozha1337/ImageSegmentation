% ������ ���������
Size = 300;
p = 3;
k = 3;

% ��������� ��� vMF
kappas = repmat(10, 1, k);
kappas_restore = kappas;
mus = [[-0.57, 0.57, 0.57];[0, -1, 0];[1, 0, 0];];
mus_restore = mus;
theta1.kappa = 10;
theta1.mu = mus(1, :)';

theta2.kappa = 10;
theta2.mu = mus(2, :)';

theta3.kappa = 10;
theta3.mu = mus(3, :)';

% ���������� ��������� ������� � ������� �� 1 �� k
InitY = randi(k, [Size, Size]);
% ����� ������ 
InitY = GibbsSamplerPotts(InitY, 20, 1, k, 2, 4);
% ����� ������ �������� �� ��������������� �������
Sample = InitY(1, :);

% ���������� ������� �� VMF � ��������� �����������
D = vmffactory(p);
clust1 = D.sample(theta1, sum(Sample==1));
clust2 = D.sample(theta2, sum(Sample==2));
clust3 = D.sample(theta3, sum(Sample==3));

% �������� ��������������� �������� � ��������������� ��� ��� �����
data = zeros([Size * Size, p]);
data(Sample==1, :) = clust1';
data(Sample==2, :) = clust2';
data(Sample==3, :) = clust3';

% ��������� ������ �� �������� ��������� ��� ������������
% mdata = mean(data, 2);
% imagesc(reshape(Sample, [Size, Size])); figure(); imagesc(reshape(mdata, [Size, Size]));

% ������ ��������� mu �������� �� ���, � �������� �������������� �������
mus = [[0, 0, 1];[0, 1, 0];[-1, 0, 0]];
for i=1:5
    % ������� �����������
    [probs, logprobs] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % �������� MLE
    segment_init = MLE(data, probs);
    % �������������� ����������� � ��������� �������
    segment_init = reshape(segment_init, [Size, Size]);
    % ���� MAP-������
    %[map, energy] = MRF_MAP_GraphCutABSwap(segment_init, logprobs, 2, k, 10, 4);
    [map, energy] = MRF_MAP_GraphCutAExpansion(segment_init, logprobs, 2, k, 10, 4);
    % ����������� ���������
    [~, mus, kappas] = EstimateParametersOneSample(data, energy, k, p, 2, mus, kappas);
end
% ��������� ����������� mu, kappa � ��������� �����������
mus_em = mus;
kappas_em = kappas;
final_segm_hmrf_em = map;

% ���������� ������������ kappa � ������������� mu
kappas = kappas_restore;
mus = [[0, 0, 1];[0, 1, 0];[-1, 0, 0]];
for i=1:5
    % ������� �����������
    [probs, ~] = CalculateLikelihoodProbabilities(data, k, kappas, mus);
    % �������� ��������� �����������
    segment_init = MLE(data, probs);
    % �������������� ����������� � ��������� �������
    segment_init = reshape(segment_init, [Size, Size]);
    % ���� MAP-������ 
    [samples] = GibbsSamplerVMF(data, segment_init, 10, 10, k, p, 2, mus, kappas, 4);
    % ������������ ���������
    [~, mus, kappas] = EstimateParameters(data, samples, k, p, 2, mus, kappas);
end
% ��������� ����������� mu, kappa � ��������� �����������
mus_mcem = mus;
kappas_mcem = kappas;
final_segm_hmrf_mcem = reshape(samples(end, :), [Size, Size]);

% ����������� GroundTruth � �����������
imagesc(reshape(Sample, [Size, Size]));
figure();
imagesc(final_segm_hmrf_em);
figure();
imagesc(final_segm_hmrf_mcem);

