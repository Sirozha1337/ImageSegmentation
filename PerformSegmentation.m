% Производит сегментацию изображения на k классов
% X - исходные данные
% k - количество классов
% max_iter - максимальное количество итераций
% burn_in - количество пропускаемых значений цепи при сэмплинге Гиббса
% sample_num - количество значений цепи применяемых для оценки параметров
% Возвращает трехмерную матрицу Y, размеченную на k классов
function [Y] = PerformSegmentation(X, k, max_iter, burn_in, sample_num, method)

p = size(X, 4);
X = double(X);
Xnorm = reshape(X, [size(X,1)*size(X,2)*size(X,3), p]);
%Xnorm = NormalizeToZeroMeanUnitStd(Xnorm);
Xnorm = NormalizeToUnitLength(Xnorm);
Xnorm = Xnorm - mean(Xnorm(:));

Yinit = kmeans(Xnorm, k, 'MaxIter', 500);

vmf = vmffactory(p);
mus = zeros(k, p);
kappas = zeros(1, k);
beta = 1/2;
for l=1:k
    roi = Xnorm(Yinit==l, :);
    [theta] = vmf.estimatedefault(roi');
    mus(l, :) = squeeze(theta.mu);
    kappas(l) = theta.kappa;
end

Yinit = reshape(Yinit, [size(X, 1), size(X, 2), size(X,3)]);
Xnorm = reshape(Xnorm, size(X));
for i=1:max_iter
    sample = GibbsSamplerVMF(Xnorm, Yinit, burn_in, sample_num, k, p, beta, mus, kappas);
    [beta, mus, kappas] = EstimateParameters(Xnorm, sample, k, p, beta, mus, kappas);
    sample = reshape(sample, [sample_num, size(Yinit, 1), size(Yinit, 2), size(Yinit,3)]);
    Yinit = squeeze(sample(sample_num,:,:,:));
end

Y = Yinit;
