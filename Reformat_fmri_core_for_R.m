clear;
ds = load('data-starplus-04847-v7.mat');
data = ds.data;
meta = ds.meta;
dt = data{1};
img = zeros([64,64,8,size(dt, 1)]);

for t = 1:size(dt, 1)
    for i = 1:size(dt, 2)
       coord = meta.colToCoord(i, :);
       roi = cell2mat(meta.colToROI(i, :));
       img(coord(1), coord(2), coord(3), t) = dt(t, i);
    end
end
save('RProject/data/data.mat', 'img')

for l = 1:length(meta.rois)
   img = zeros([64,64,8]);
   for i = 1:size(dt, 2)
        coord = meta.colToCoord(i, :);
        roi = cell2mat(meta.colToROI(i, :));
        img(coord(1), coord(2), coord(3)) = strcmp(roi, meta.rois(l).name);
   end
   save(strcat('RProject/data/', meta.rois(l).name), 'img');
end