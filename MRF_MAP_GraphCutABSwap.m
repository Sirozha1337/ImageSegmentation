%%  Алгоритм нахождения MAP-оценки
%---input---------------------------------------------------------
%   X: исходное разбиение, матрица, каждая ячейка содержит значение от 1:L
%   logprobs: отрицательный логарифм функции правдоподобия, матрица LxN
%   b: параметр модели Поттса, скаляр либо матрица NCxN, 
%   где NC - количество соседей, N - количество пикселей
%   L: количество меток
%   MAP_iter: максимальное количество итераций
%   neighbours_count: количество соседей, доступные значения
%   2-D: 4, 8, 16
%   3-D: 6, 26
%---output--------------------------------------------------------
%   X: финальная сегментация
%   posterior: постериорная вероятность финальной сегментации
function [X, posterior]=MRF_MAP_GraphCutABSwap(X,logprobs,b,L,MAP_iter,neighbours_count)
% reshape x
sz = size(X);
flat = X(:);
flatsize = size(flat, 1);
% neighbours indexes
all_neighbours_ind = GetNeighbours(sz, neighbours_count);

all_neighbours = flat(all_neighbours_ind);
        
% create two new vertices
terminal0 = flatsize + 1;
terminal1 = flatsize + 2;

minimum_U = Inf;

for i=1:MAP_iter
    success = 0;
    fprintf('\tInner Iteration: %d\n',i);
    fprintf('\tCurrent U: %d\n',minimum_U);
    abcomb = combnk(1:L, 2);
    %permutations = randperm(size(abcomb, 1));
    for j=1:size(abcomb,1)%permutations
        alpha = abcomb(j, 1);
        beta = abcomb(j, 2);
        
        % vector of vertex indexes that are labeled as alpha or beta
        ind = find(flat == alpha | flat == beta);
        ind_size = size(ind, 1);
        if ind_size > 0
            % construct edges
            s = [ind; ...
                 ind; ...
                 transpose(reshape(repmat(ind, neighbours_count, 1).',1,[]))];


            t = [repmat(terminal0, ind_size, 1) .* (flat(ind) == alpha | flat(ind) == beta); ...
                 repmat(terminal1, ind_size, 1) .* (flat(ind) == alpha | flat(ind) == beta); ...
                 reshape((all_neighbours_ind(:, ind) .* (flat(all_neighbours_ind(:, ind)) == alpha | flat(all_neighbours_ind(:, ind)) == beta)).',1,[])'];


            % calculate likelihood probabilities
            alpha_prob = logprobs(alpha, ind);
            beta_prob = logprobs(beta, ind);
            
            % construct vector of weights
            weights = [ (alpha_prob + b * sum((all_neighbours(:, ind) ~= alpha) .* (all_neighbours(:, ind) ~= beta)))'; ...
                        (beta_prob  + b * sum((all_neighbours(:, ind) ~= alpha) .* (all_neighbours(:, ind) ~= beta)))'; ...
                        b * reshape((all_neighbours(:, ind)' ~= repmat(flat(ind), 1, neighbours_count)).', 1, [])'];
                    
            % remove duplicated edges
            non_empty_links = find(t~=0); 
            s = s(non_empty_links);
            t = t(non_empty_links);
            non_self_ref_links = find(t~=s);
            s = s(non_self_ref_links);
            t = t(non_self_ref_links);
            combo = [s t];
            [~, uniq_ind, ~] = unique(sort(combo,2), 'rows');
            s = s(uniq_ind);
            t = t(uniq_ind);
            % add min weight only to the t-links
            min_weight = min(weights(1:(numel(ind)*2)));
            if min_weight < 0
                fprintf('\t\tSome weight is lower than zero: %f\n', min_weight);
                weights(1:(numel(ind)*2)) = weights(1:(numel(ind)*2)) - min_weight;
            end
            weights = weights(non_empty_links);
            weights = weights(non_self_ref_links);
            weights = weights(uniq_ind);

            % create graph from edges and weights
            G = graph(s, t, weights);
            % calculate max flow
            [sum_U,~,cs,ct] = maxflow(G,terminal0,terminal1);

            if sum_U < minimum_U
                fprintf('\tSetting new U: %d\n',sum_U);
                minimum_U = sum_U;
                % update image according to maxflow list of vertices
                cs = cs(cs < terminal0);
                ct = ct(ct < terminal0);
                cs = cs(ismember(cs, ind));
                ct = ct(ismember(ct, ind));
                flat(ct) = alpha;
                flat(cs) = beta;
                X(ct) = alpha;
                X(cs) = beta;
                success = 1;
            end
        end
    end
    if success == 0
        break;
    end
end

posterior = zeros(L, flatsize);
for i=1:L
   posterior(i, :) = exp(-logprobs(i, :) - b * sum(all_neighbours ~= i));
   posterior(i, X==0) = 0;
end
norm_const = sum(posterior, 1);
posterior = bsxfun(@rdivide,posterior,norm_const);
