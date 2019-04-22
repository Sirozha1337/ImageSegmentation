%%  Алгоритм генерации синтетических данных из распределения von Mises-Fisher
%---input---------------------------------------------------------
%   init: начальная конфигурация
%   p: количество измерений
%   k: количество меток
%   beta: параметр модели Поттса
%   mus: параметр распределения vMF
%   kappas: параметр распределения vMF
%   burn_in: количество отбрасываемых значений схемы Гиббса
%   neighbours_count: количество соседей, доступные значения
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   data: данные из распределения vMF
%   gt: ground truth для сгенерированных данных
function [data, gt]=GenerateSynteticData(init, p, k, beta, mus, kappas, burn_in, neighbours_count)

% Схема Гиббса 
gt = GibbsSamplerPotts(init, burn_in, 1, k, beta, neighbours_count);
% Берем первую картинку из сгенерированной выборки
gt = gt(1, :);

% параметры для vMF
theta = cell([k, 1]);
for i=1:k
    theta{i}.kappa = kappas(i);
    theta{i}.mu = mus(i, :)';
end

D = vmffactory(p);
data = zeros([size(init(:), 1), p]);
for i=1:k
    % Генерируем выборку из VMF с заданными параметрами
    clust = D.sample(theta{i}, sum(gt==i));
    % Собираем сгенерированные значения в предназначенные для них места
    data(gt==i, :) = clust';
end