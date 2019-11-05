function [val] = C(p, kappa)
val = min(kappa^(p/2-1), 10^100)/min(((2*pi)^(p/2)*besseli(p/2-1, kappa)), 10^100);