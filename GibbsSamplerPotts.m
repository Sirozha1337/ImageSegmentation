% √енерирует выборку из распределени€ √иббса
% X - матрица начальной конфигурации
% B - количество пропускаемых состо€ний
% M - возвращаемое количество состо€ний
% k - количество классов сегментации
% beta - параметр модели ѕоттса
% neighbours_count - количество соседей
% ¬озвращает матрицу размерности: Mx(количество вокселей)
function [Y] = GibbsSamplerPotts(X, B, M, k, beta, neighbours_count)

sz = size(X);
flat = X(:);
flatsz = size(flat,1);
all_neighbours_ind = GetNeighbours(sz, neighbours_count);
Y = zeros(M, flatsz);
for j=1:(B+M)
    permutations = randperm(flatsz);
    for i=permutations
       P = zeros(1, k);
       pex = zeros(1, k);
       for l=1:k
           neighbours = all_neighbours_ind(all_neighbours_ind(:, i)~=i, i);
           neib = -beta * sum(flat(neighbours)~=l);
           pex(l) = exp(neib);
       end
       for l=1:k
           P(l) = pex(l)/sum(pex);
       end
       
       if sum(P.^2) > 0
           ind = randsample(1:k, 1, true, P);
           flat(i) = ind;
       end
    end
    if j > B
        Y(j-B, :) = flat;
    end
end