function [Y] = MLE(X, probs)

Y = ones([size(X,1),1]);
sz = size(probs, 1);
prev_prob = probs(1, :);
for i=2:sz
    Y(probs(i, :) > prev_prob) = i;
    prev_prob = max(probs(i), prev_prob);
end