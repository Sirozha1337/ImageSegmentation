% CALC, LFEF, RFEF, LSGA, RSGA, LSPL, RSPL, LIT, RIT - visual
% LDLPFC, RDLPFC, LOPER, ROPER - cognition, memory, decision making
% LIFG, LPPREC, RPPREC, LTRIA, RTRIA, LIPL, RIPL - speech
% LT, RT - hearing 
% SMA, LIPS, RIPS - motor
function [data] = ReadROIFromStarplus(subject, roi_name)

if contains(roi_name, "_")
    roi_names = strsplit(roi_name, "_");
else
    roi_names = roi_name;
end

data = zeros([64,64,8]);
ds = load(sprintf('fmri_core/data-starplus-0%d-v7.mat', subject));
for j = 1:size(roi_names, 2)
    cur_roi = roi_names(j);
    meta = ds.meta;
    for i = 1:size(meta.colToCoord, 1)
        coord = meta.colToCoord(i, :);
        roi = cell2mat(meta.colToROI(i, :));
        data(coord(1), coord(2), coord(3)) = data(coord(1), coord(2), coord(3)) | strcmp(roi, cur_roi);
    end
end