function [val] = logC(p, kappa)
b1 = min(besseli((1/2)*p, kappa, 1), 10^300);
b2 = min(besseli((1/2)*p+1, kappa, 1), 10^300);
val = (1/2)*p*log(kappa)-(1/2)*p*log(2)-(1/2)*p*log(pi)-log(p*b1+b2*kappa)-kappa;