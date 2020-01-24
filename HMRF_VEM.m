%%  �������� ���������� MAP-������
%---input---------------------------------------------------------
%   data: �������� ������, ������� XxYxZxP
%   p: ��������� ����������� (����� ��� time series)
%   k: ���������� �������
%   beta: �������� ������ ������
%   mus: ��������� ��� ������-������
%   kappas: ��������� ��� ������-������
%   lambda: �������� Mean Field
%   MAP_iter: ������������ ���������� ��������
%   INNER_iter: ������������ ���������� ���������� ��������
%   neighbours_count: ���������� �������, ��������� ��������
%   2-D: 4, 8, 16
%   3-D: 6, 26
%   mask: ����� ��� �����������, ������� XxYxZ
%   1 ��� ����� ������������ ����������
%   0 ����� ������������
%---output--------------------------------------------------------
%   sample: ������� ��������� �����������
%   beta: ��������� �������� beta
%   kappas: ��������� ��� ������-������
%   mus: ��������� ��� ������-������
function [sample, beta, mus, kappas, sample2]=HMRF_VEM(data, p, k, beta, mus, kappas, ...
                                        lambda, MAP_iter, INNER_iter, ...
                                        neighbours_count, mask)

sz = size(data);
flatsize = prod(sz(1:end-1));
flat = reshape(data, [flatsize, p]);

if(nargin < 11)
    mask = ones(sz(1:end-1));
end

% neighbours indexes
all_neighbours_ind = GetNeighbours(sz(1:end-1), neighbours_count);

for i=1:MAP_iter
    fprintf('\tIteration: %d out of %d\n',i,MAP_iter);
    [~, logprobs] = CalculateLikelihoodProbabilities(flat, k, kappas, mus, mask);
    Q = zeros(flatsize, k);
    Q(:, 1) = ones(flatsize, 1);
    for u=1:INNER_iter
        Qtilde = exp(beta ...
            * squeeze(sum(reshape(Q(all_neighbours_ind, :), [neighbours_count, flatsize, k]), 1)) ...
            - logprobs');
        Qtilde = min(Qtilde, 10^100);
        sumQ = sum(Qtilde, 2);
        Q = (1-lambda) * Q + lambda * Qtilde ./ sumQ;
        Q(mask==0, :) = 0;
    end
    for l=1:k
        R = Q(mask~=0, l)'*flat(mask~=0,:);
        Rlen = sqrt(sum(R .^ 2));
        if Rlen ~= 0
            mus(l, :) = R / Rlen;
        else
            fprintf('WARNING: Rlen is zero\n');
        end
        
        eqn = @(x) besseli(p/2, x)/besseli(p/2-1,x) * sum(Q(mask~=0, l)) - Rlen;
        initval = kappas(l);
        if isnan(eqn(initval))
            fprintf('WARNING: function is NaN at initial value\n');
            initval = 10;
        end
        opts1 =  optimset('display','off');
        kappas(l) = lsqnonlin(eqn, initval, 0, Inf, opts1);
    end
end

% Find MAP
[~, sample] = max(Q, [], 2);
sample = reshape(sample, sz(1:end-1));
sample(mask==0) = 0;
[~, logprobs] = CalculateLikelihoodProbabilities(flat, k, kappas, mus, sample);
sample2 = MRF_MAP_GraphCutAExpansion(sample, logprobs, beta, k, MAP_iter, neighbours_count);