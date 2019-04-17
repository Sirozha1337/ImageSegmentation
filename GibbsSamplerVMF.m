%% Генерирует выборку из распределения Гиббса
% X - 4-х мерная матрица исходных данных
% Yinit - 3-х мерная матрица начальной сегментации
% B - количество пропускаемых состояний
% M - возвращаемое количество состояний
% k - количество классов сегментации
% p - размерность данных (количество time-series)
% beta - параметр модели Поттса
% mu - матрица параметров для vMF
% kappa - вектор параметров для vMF
% neighbours_count - количество соседей
% Возвращает матрицу размерности: Mx(количество вокселей)
function [Y] = GibbsSamplerVMF(X, Yinit, B, M, k, p, beta, mu, kappa, neighbours_count)

sz = size(Yinit);
Yflat = Yinit(:);
flatsz = size(Yflat,1);
all_neighbours_ind = GetNeighbours(sz, neighbours_count);
Y = zeros(M, flatsz);
for j=1:(B+M)
    permutations = randperm(flatsz);
    for i=permutations
       P = zeros(1, k);
       pex = zeros(1, k);
       for l=1:k
           neighbours = all_neighbours_ind(all_neighbours_ind(:, i)~=i, i);
           neib = -beta * sum(Yflat(neighbours)~=l);
           lik = kappa(l) * mu(l, :) * X(i, :)'-log(C(p, kappa(l)));
           pex(l) = exp(neib + lik);
       end
       for l=1:k
           P(l) = pex(l)/sum(pex);
       end
       
       if sum(P.^2) > 0
           ind = randsample(1:k, 1, true, P);
           Yflat(i) = ind;
       end
    end
    if j > B
        Y(j-B, :) = Yflat;
    end
end