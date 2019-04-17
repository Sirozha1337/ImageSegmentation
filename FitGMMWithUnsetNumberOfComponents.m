function [gmm]=FitGMMWithUnsetNumberOfComponents(data, maxNum)

AIC = zeros(1,maxNum);
GMModels = cell(1, maxNum);

for k=1:maxNum
    GMModels{k} = fitgmdist(data,k);
    AIC(k)= GMModels{k}.AIC;
end

[~,numComponents] = min(AIC);

gmm = GMModels{numComponents};