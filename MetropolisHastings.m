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
function [Y, sample] = MetropolisHastings(Yinit, B, M, k, beta, logprobs, neighbours_count, labelCosts)

sz = size(Yinit);
Yflat = Yinit(:);
flatsz = size(Yflat,1);
all_neighbours_ind = GetNeighbours(sz, neighbours_count);
Y = zeros(M, flatsz);
if nargin >= 8
    counts = zeros(1,k);
    for i=1:k
        counts(i) = sum(Yflat==k);
    end
end

for j=1:(B+M)
    permutations = randperm(flatsz);
    non_zero_perms = permutations(Yinit(permutations)~=0);
    for i=non_zero_perms
        neighbours = all_neighbours_ind(all_neighbours_ind(:, i)~=i, i);
        
        cur_state = Yflat(i);
        
        new_state = randsample(k, 1);
        
        if nargin < 8
            cur_energy = -beta * sum(Yflat(neighbours)~=cur_state, 1) - logprobs(cur_state, i);
            new_energy = -beta * sum(Yflat(neighbours)~=new_state, 1) - logprobs(new_state, i);
        else
            new_counts = counts + [zeros(1, cur_state-1), -1, zeros(1, k-cur_state)]  + [zeros(1, new_state-1), 1, zeros(1, k-new_state)];
            cur_energy = -beta * sum(Yflat(neighbours)~=cur_state, 1) - logprobs(cur_state, i) + sum(labelCosts(:) .* (counts(:)>0));
            new_energy = -beta * sum(Yflat(neighbours)~=new_state, 1) - logprobs(new_state, i) + sum(labelCosts(:) .* (new_counts(:)>0));
        end
        P = min(exp(new_energy - cur_energy), 1);

        if rand() < P
            Yflat(i) = new_state;
            if nargin >= 8
                counts = new_counts;
            end
        end
    end
    if j > B
        Y(j-B, :) = Yflat;
    end
end
sample = reshape(Y(end, :), sz);