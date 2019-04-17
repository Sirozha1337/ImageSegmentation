%% ���������� ������� �� ������������� ������
% X - 4-� ������ ������� �������� ������
% Yinit - 3-� ������ ������� ��������� �����������
% B - ���������� ������������ ���������
% M - ������������ ���������� ���������
% k - ���������� ������� �����������
% p - ����������� ������ (���������� time-series)
% beta - �������� ������ ������
% mu - ������� ���������� ��� vMF
% kappa - ������ ���������� ��� vMF
% neighbours_count - ���������� �������
% ���������� ������� �����������: Mx(���������� ��������)
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