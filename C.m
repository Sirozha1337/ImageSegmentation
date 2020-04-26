function [val] = C(p, kappa)
val = exp((p/2-1)*log(kappa) - log((2*pi)^(p/2)) - log(besseli(p/2-1, kappa, 1)) - kappa);