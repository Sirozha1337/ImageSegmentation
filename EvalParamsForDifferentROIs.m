% читаем данные
data = Read4DArrayFromStarplus(4847, 1);
data2 = Read4DArrayFromStarplus(4799, 1);
roi_names = ["CALC", "LIPL", "LDLPFC", "LIPS", "LOPER", "LT", "LTRIA"];

% раскладываем данные по рядам и нормализуем
timeseries_by_rows = reshape(data, [size(data, 1) * size(data, 2) * size(data, 3), size(data, 4)]);
timeseries_by_rows = double(timeseries_by_rows);
timeseries_by_rows = NormalizeToUnitLength(timeseries_by_rows);
timeseries_by_rows2 = reshape(data2, [size(data2, 1) * size(data2, 2) * size(data2, 3), size(data2, 4)]);
timeseries_by_rows2 = double(timeseries_by_rows2);
timeseries_by_rows2 = NormalizeToUnitLength(timeseries_by_rows2);

% инициализируем структуры для параметров распределения
vmf = vmffactory(size(data, 4));
mus = zeros(size(roi_names, 2), size(data, 4));
kappas = zeros(1, size(roi_names, 2));
vmf2 = vmffactory(size(data2, 4));
mus2 = zeros(size(roi_names, 2), size(data2, 4));
kappas2 = zeros(1, size(roi_names, 2));
for i=1:size(roi_names, 2)
    roi_name = roi_names(i);
    % достаем данные соответствующие размеченной области
    roi_mask = ReadROIFromStarplus(4847, roi_name);
    roi_mask = roi_mask(:);
    roi = timeseries_by_rows(roi_mask==1, :);
    [theta] = vmf.estimatedefault(roi');
    mus(i, :) = squeeze(theta.mu);
    kappas(i) = theta.kappa;
    % достаем данные соответствующие размеченной области
    roi_mask = ReadROIFromStarplus(4799, roi_name);
    roi_mask = roi_mask(:);
    roi = timeseries_by_rows2(roi_mask==1, :);
    [theta] = vmf2.estimatedefault(roi');
    mus2(i, :) = squeeze(theta.mu);
    kappas2(i) = theta.kappa;
end

