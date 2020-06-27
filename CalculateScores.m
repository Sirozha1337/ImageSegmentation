function[simple_score, jaccard_score, ssim_score] = CalculateScores(gt, segm, k)
    [simple_score, best_segm] = SimpleSimilarityScore(gt, segm, k);
    [jaccard_score, best_segm2] = SimilarityScore(gt, best_segm, k);
    sz = size(gt);
    ssim_score1 = ssim(reshape(best_segm,sz), reshape(gt,sz));
    ssim_score2 = ssim(reshape(best_segm2,sz), reshape(gt,sz));
    ssim_score = max([ssim_score1,ssim_score2]);
end