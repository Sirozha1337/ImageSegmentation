# ImageSegmentation
Matlab Image Segmentation scripts

## Required packages
* SPM12: https://www.fil.ion.ucl.ac.uk/spm/software/spm12/
* MixEst: https://github.com/utvisionlab/mixest
* GCO (used only in TestGCO.m): https://vision.cs.uwaterloo.ca/code/

## Experiments
### FMRI
* **prepros_nyu_data.R** -- preprocess NYU fmri data (https://www.nitrc.org/projects/nyu_trt)
* **fmriTestScript** -- runs HMRF-MCEM, HMRF-VEM, HMRF-EM, GrabCut on preprocessed NYU fmri data
### Synthetic data
* **MultipleIterationsOfTheExperiment** -- generates synthetic data and runs HMRF-MCEM, HMRF-VEM, HMRF-EM, GrabCut on that data
* **CalculateMetricsForExperiments** -- calculates metrics for synthetic data experiments and saves them to csv
* **AnalyzeExperimentMetrics** -- reads calculated metrics and displays boxplots
### Other
* **CompareMethodsBinarySegmentation** -- segments 2D image using ICM, Simulated Annealing and GraphCuts
* **CompareMethodsNonBinarySegmentation** -- segments 2D image into 3 classes using ICM, Simulated Annealing and GraphCuts
* **CompareTime** -- compares the time it takes to run HMRF-MCEM and HMRF-VEM
* **GraphCutOnStarplusData** -- attempt to segment Starplus fmri data
* **TestGCO** -- an example of using GCO library to segment an image with label costs

## Functions
* **AUC** -- calculates AUC metric for a given ground truth and posterior distribution
* **C** -- vMF normalization const
* **logC** -- log vMF normalization const
* **CalculateFinalEnergy** -- calculates energy of an image 
* **CalculateLikelihoodProbabilities** -- calculates likelihood probabilities for vMF distribution
* **CalculateScores** -- calculates Simple Similarity Score, Jaccard Coefficient and SSIM for a given image and ground truth
* **EstimateParametersGrabCut** -- estimates parameters for GrabCut algorithm
* **EstimateParametersHMRFEM** -- estimates parameters for HMRF-EM algorithm
* **EstimateParametersHMRFMCEM** -- estimates parameters for HMRF-MCEM algorithm
* **FitGMMWithUnsetNumberOfComponents** -- attempts to fit gmms with variable number of components and chooses best by AIC
* **GenerateSynteticData** -- generates syntetic data using Potts model and von Mises Fisher distribution
* **GetNeighbours** -- provides a list of neighbour indexes for a list of points
* **GibbsSamplerPotts** -- Gibbs sampler for Potts model
* **GibbsSamplerLabelCost** -- Gibbs sampler with a possibility to provide label costs
* **Grab_Cut** -- an implementation of GrabCut algorithm
* **HMRF_EM** -- an implementation of HMRF-EM algorithm
* **HMRF_MCEM** -- an implementation of HMRF-MCEM algorithm
* **HMRF_VEM** -- an implementation of HMRF-VEM algorithm
* **ICM** -- iterated conditional modes for MRF
* **MetropolisHastings** -- an implementation of Metropolis Hastings algorithm 
* **MLE** -- calculates segmentation based maximum likelihood probability
* **MRF_MAP_GraphCutABSwap** -- an implementation of GraphCut Alpha-Beta Swap algorithm
* **MRF_MAP_GraphCutAExpansion** -- an implementation of GraphCut Alpha Expansion algorithm
* **NormalizeToUnitLength** -- normalizes vectors to N-dimensional unit sphere
* **PadData** -- adds zero padding to the matrix
* **PlotDistanceToTruth** -- plots the difference between true values and estimated values
* **RealignDataByMask** -- applies mask to data and removes zeroes on the edges of the data
* **RotateMatrix** -- rotates matrix 90 degress around a given dimension
* **SaveImage** -- saves segmented image to file using same colors as imagesc
* **ShowImageWithLabels** -- shows slice of 3d dimensional image overlayed with its labels
* **ShowMultipleSlicesWithLabels** -- shows multiple slices of 3d dimensional image overlayed with its labels
* **SimilarityScore** -- calculates Jaccard similarity score
* **SimpleSimilarityScore** -- calcultates simple similarity score
* **SimulatedAnnealing** -- an implementation of Simulated Annealing for MRF
* **TruePositiveNegativeRates** -- calculate sensitivity and recall metrics
* **Read4DArrayFromStarplus** -- reads 4D array from Starplus fmri data (http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-81/www/)
* **ReadROIFromStarplus** -- reads provided ROI from Starplus fmri data

