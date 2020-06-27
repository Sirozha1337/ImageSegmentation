library(neurobase)
library(matlabr)
library(spm12r)

files = c(anatomical = "sub38579/anat/mprage_skullstripped.nii.gz",
          functional = "sub38579/func/lfo.nii.gz")

####################################
# Realignment
####################################
if (have_matlab()) {
  realigned = spm12_realign( 
    filename = files["functional"], 
    register_to = "mean",
    reslice = "mean",
    clean = FALSE
  )
  print(realigned)
  mean_img = realigned[["mean"]]
  mean_nifti = readnii(mean_img)
}

####################################
# Anatomical MRI Coregistration to Mean fMRI
####################################
if (have_matlab()) {
  anatomical = files["anatomical"]
  anat_img = checknii(anatomical)
  print(anat_img)
  acpc_reorient(
    infiles = anat_img,
    modality = "T1")
  
  coreg = spm12_coregister(
    fixed = mean_img,
    moving = anat_img,
    prefix = "r")
  
  coreg_anat = coreg$outfile
  coreg_img = readnii(coreg_anat)
  double_ortho(coreg_img, mean_nifti)
}

####################################
# Anatomical MRI Segmentation
####################################
if (have_matlab()) {
  seg_res = spm12_segment(
    filename = coreg_anat,
    set_origin = FALSE,
    retimg = FALSE)
  print(seg_res)
}

####################################
# Spatial Normalization Transformation
####################################
bbox = matrix(
  c(-90, -126, -72, 
    90, 90, 108), 
  nrow = 2, byrow = TRUE)
if (have_matlab()) {
  norm = spm12_normalize_write(
    deformation = seg_res$deformation,
    other.files = c(coreg_anat, mean_img, realigned$outfiles),
    bounding_box = bbox,
    retimg = FALSE, 
    clean = FALSE)
  print(norm)
  norm_data = norm$outfiles
  names(norm_data) = c("anat", "mean", "fmri")
  norm_mean_img = readnii(norm_data["mean"])
  norm_anat_img = readnii(norm_data["anat"])
  double_ortho(norm_mean_img, norm_anat_img)
}

####################################
# Anatomical MRI Segmentation
####################################
if (have_matlab()) {
  seg_res = spm12_segment(
    filename = norm_anat_img,
    set_origin = FALSE,
    retimg = FALSE)
  
  alpha = function(col, alpha = 1) {
    cols = t(col2rgb(col, alpha = FALSE)/255)
    rgb(cols, alpha = alpha)
  } 
  
  seg_files = check_nifti(seg_res$outfiles)
  hard_seg = spm_probs_to_seg(seg_files)
  hard_seg[ hard_seg > 1] = 0
  
  ortho2(norm_anat_img, hard_seg, 
         col.y = alpha(c("red", "green", "blue"), 0.5))
  
  norm_fmri_img = readnii(norm_data["fmri"])
  ortho2(norm_fmri_img[,,,15], hard_seg, 
         col.y = alpha(c("red", "green", "blue"), 0.5))
}

writeNIfTI(norm_fmri_img, "sub38579/prepros/fmri")
writeNIfTI(norm_anat_img, "sub38579/prepros/anat")
writeNIfTI(norm_mean_img, "sub38579/prepros/mean")
writeNIfTI(hard_seg, "sub38579/prepros/gray_matter")
