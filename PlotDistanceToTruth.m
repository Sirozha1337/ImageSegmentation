function [fig] = PlotDistanceToTruth(true_values, estimated_values, vector, label_text, title_text)

fig = figure();
for j=1:size(estimated_values, 2)
    dist = zeros([1, size(estimated_values, 1)]);
    for i=1:size(estimated_values, 1)
        if vector == 1
            dist(i, :) = acos(dot(squeeze(true_values(j,:)), squeeze(estimated_values(i, j, :))));
            %dist(i, :) = norm(squeeze(true_values(j,:))-squeeze(estimated_values(i, j, :)));
        else
            dist(i, :) = norm(squeeze(true_values(j))-squeeze(estimated_values(i, j)));
        end
    end
    plot(1:size(estimated_values, 1), dist);
    hold on;
end
title(title_text);
xlabel('Iterations')
ylabel(label_text) 
hold off;