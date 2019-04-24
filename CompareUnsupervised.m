% Задаем параметры
Size = 300;
p = 3;
k = 3;
kappas = repmat(10, 1, k);
mus = [[-0.57, 0.57, 0.57];[-0.6305, 0.3152, 0.7093]; [-0.9879, 0.1098, 0.1098];];
close_mus = [[-0.1231, 0.8616, 0.4924];[-0.7049, 0.0783, 0.7049]; [-0.9847, 0.1231, 0.1231];];
close_kappas = [5, 8, 7];

% генерируем случайную матрицу с числами от 1 до k
InitY = randi(k, [Size, Size]);
[data, gt]=GenerateSynteticData(InitY, p, k, 2, mus, kappas, 20, 4);
imwrite(reshape(data, [Size, Size, 3]), 'data/generated.png');

% MCEM
[final_segm_mcem, beta_mcem, mus_mcem, kappas_mcem] = HMRF_MCEM(data, [Size, Size], k, 2, close_mus, close_kappas, 10, 10, 10, 4);
[final_segm_grc, mus_grc, kappas_grc] = Grab_Cut(data, InitY, k, 10, 40, close_mus, close_kappas, 5, 10, 4);
final_segm_kmeans = kmeans(data, k);
[final_segm_expansion, beta_expansion, mus_expansion, kappas_expansion] = HMRF_EM(data, [Size, Size], k, 2, close_mus, close_kappas, 10, 10, 4, 'expansion');
[final_segm_swap, beta_swap, mus_swap, kappas_swap] = HMRF_EM(data, [Size, Size], k, 2, close_mus, close_kappas, 10, 10, 4, 'swap');

%imwrite( ind2rgb(im2uint8(mat2gray(final_segm_grc)), parula(256)), 'data/generated2.png')

dsc_mcem = SimilarityScore(gt, final_segm_mcem, k);
dsc_grc = SimilarityScore(gt, final_segm_grc, k);
dsc_expansion = SimilarityScore(gt, final_segm_expansion, k);
dsc_swap = SimilarityScore(gt, final_segm_swap, k);
dsc_kmeans = SimilarityScore(gt, final_segm_kmeans, k);

%imagesc(reshape(gt, [Size,Size]));
%figure(); imagesc(reshape(final_segm_grc, [Size,Size]));
%figure(); imagesc(reshape(final_segm_mcem, [Size,Size]));