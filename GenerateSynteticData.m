% ������ ���������
Size = 300;
p = 3;
k = 3;

% ��������� ��� vMF
kappas = repmat(10, 1, k);
kappas_restore = kappas;
mus = [[-0.57, 0.57, 0.57];[0, -1, 0]; [1, 0, 0];];
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
%mus = [[0, 0, 1];[0, 1, 0];[-1, 0, 0]];

[final_segm_hmrf_em, beta_em, mus_em, kappas_em] = HMRF_EM(data, [Size, Size], k, 2, mus, kappas, 10, 5, 4, 'expansion');

% ���������� ������������ kappa � ������������� mu
kappas = kappas_restore;
mus = mus_restore;
% ��������� ����������� mu, kappa � ��������� �����������
[final_segm_hmrf_mcem, beta_mcem, mus_mcem, kappas_mcem] = HMRF_MCEM(data, [Size, Size], k, 2, mus, kappas, 10, 10, 5, 4);

% ����������� GroundTruth � �����������
imagesc(reshape(Sample, [Size, Size]));
figure();
imagesc(final_segm_hmrf_em);
figure();
imagesc(final_segm_hmrf_mcem);

dsc_em = SimilarityScore(Sample, final_segm_hmrf_em, k);
dsc_mcem = SimilarityScore(Sample, final_segm_hmrf_mcem, k);

