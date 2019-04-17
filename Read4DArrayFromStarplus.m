function [data] = Read4DArrayFromStarplus(subject, obs_num)

ds = load(sprintf('fmri_core/data-starplus-0%d-v7.mat', subject));
meta = ds.meta;
dt = ds.data{obs_num};
data = zeros([64, 64, 8, size(dt, 1)]);

for t = 1:size(dt, 1)
    for i = 1:size(dt, 2)
       coord = meta.colToCoord(i, :);
       data(coord(1), coord(2), coord(3), t) = dt(t, i);
    end
end

