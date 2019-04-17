%%  Алгоритм нахождения MAP-оценки
%---input---------------------------------------------------------
%   X: исходное разбиение, матрица, каждая ячейка содержит значение от 1:L
%   logprobs: отрицательный логарифм функции правдоподобия, матрица LxN
%   b: параметр модели Поттса
%   L: количество меток
%   MAP_iter: максимальное количество итераций
%---output--------------------------------------------------------
%   X: финальная сегментация
%   posterior: постериорная вероятность финальной сегментации
function [X, posterior]=MRF_MAP_GraphCutAExpansion(X,logprobs,b,L,MAP_iter,neighbours_count)
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
    %permutations = randperm(size(abcomb, 1));
    for alpha=1:L%permutations
        % vector of vertex indexes that are labeled as alpha or beta
        ind_alpha = find(flat == alpha);
        ind_other = find(flat ~= alpha);
        
        if size(ind_other, 1) > 0
            % создаем дополнительные вершины и ребра 
            % между соседями с разными метками
            edge_diff = zeros([flatsize * 4, 2]);
            edge_same = zeros([flatsize * 4, 2]);
            for neighbours_dir=1:size(all_neighbours, 1)
                flat_diff_ind = find(flat' ~= all_neighbours(neighbours_dir, :));
                neighbours_diff_ind = all_neighbours_ind(neighbours_dir, flat_diff_ind);
                ind_start = flatsize * (neighbours_dir - 1) + 1;
                ind_end = flatsize * (neighbours_dir - 1) + size(flat_diff_ind, 2);
                edge_diff(ind_start:ind_end, :) = [flat_diff_ind' neighbours_diff_ind'];
                
                flat_same_ind = find(flat' == all_neighbours(neighbours_dir, :));
                neighbours_same_ind = all_neighbours_ind(neighbours_dir, flat_same_ind);
                ind_start = flatsize * (neighbours_dir - 1) + 1;
                ind_end = flatsize * (neighbours_dir - 1) + size(flat_same_ind, 2);
                edge_same(ind_start:ind_end, :) = [flat_same_ind' neighbours_same_ind'];
            end
            % удаляем нулевые и совпадающие ребра
            edge_diff = edge_diff(edge_diff(:, 1) ~= 0 & edge_diff(:, 2) ~= 0 & edge_diff(:, 1) ~= edge_diff(:, 2), :);
            [edge_diff, ~, ~] = unique(sort(edge_diff,2), 'rows');
            edge_same = edge_same(edge_same(:, 1) ~= 0 & edge_same(:, 2) ~= 0 & edge_same(:, 1) ~= edge_same(:, 2), :);
            [edge_same, ~, ~] = unique(sort(edge_same,2), 'rows');
            % создаем доп вершины
            a = ((terminal1+1):(terminal1+size(edge_diff, 1)))';
            
            % construct edges
            s = [ind_alpha; ...
                 ind_other; ...
                 ind_alpha; ...
                 ind_other; ...
                 edge_diff(:, 1); ...
                 edge_diff(:, 2); ...
                 a; ...
                 edge_same(:, 1)];


            t = [repmat(terminal0, size(ind_alpha)); ...
                 repmat(terminal0, size(ind_other)); ...
                 repmat(terminal1, size(ind_alpha)); ...
                 repmat(terminal1, size(ind_other)); ...
                 a; ...
                 a; ...
                 repmat(terminal1, size(a));
                 edge_same(:, 2)];

            % construct vector of weights
            weights = [ 
                        logprobs(alpha, ind_alpha)' + 10; ...
                        logprobs(alpha, ind_other)' + 10; ...
                        Inf(size(ind_alpha)); ...
                        logprobs(sub2ind(size(logprobs), flat(ind_other), ind_other)) + 10; ...
                        b * (flat(edge_diff(:, 1)) ~= alpha); ...
                        b * (flat(edge_diff(:, 2)) ~= alpha); ...
                        b * (flat(edge_diff(:, 1)) ~= flat(edge_diff(:, 2))); ...
                        b * (flat(edge_same(:, 1)) ~= flat(edge_same(:, 2))) ];
                    
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
            weights = weights(non_empty_links);
            weights = weights(non_self_ref_links);
            weights = weights(uniq_ind);

            % create graph from edges and weights
            G = graph(s, t, weights);
            % calculate max flow
            [sum_U,~,~,ct] = maxflow(G,terminal0,terminal1);

            if sum_U < minimum_U
                fprintf('\tSetting new U: %d\n',sum_U);
                minimum_U = sum_U;
                % update image according to maxflow list of vertices
                ct = ct(ct < terminal0);
                flat(ct) = alpha;
                X(ct) = alpha;
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
   posterior(i, :) = exp(-logprobs(i) + b * sum(all_neighbours ~= i));
end
norm_const = sum(posterior);
for i=1:L
    posterior(i, :) = posterior(i, :) ./ norm_const;
end
