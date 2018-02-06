# Chapter 3: Data Preprocessing

## Major Tasks
- Cleaning: Fill in missing vlaues, smooth noise, identify/remove outliers, address inconsistencies
- Integration: combine multiple sources
- Transformation: normalization/aggregation
- Reduction: reduce representation with same/similar results
- Discretization

Many ways to deal with missing data


## Discretization
- Equal-width partitioning
- Equal Depth (frequency)

## Binning Methods for Data Smoothing
Once your data has been binned (via equal-width or equal-depth), you may choose to smooth the data. Common approaches include mean (all values in the bin are replaced with the average) or boundary (first half of the values get the min boundary value, second half get the upper boundary value)

Note, similar approaches can be used for data reduction... in this case, N values may be replaced with a single that is the mean, etc.

## Data Integration

## Data Transformation

## Normalization

### Min-Max
Linear Transformation on the original data

<img src="https://latex.codecogs.com/gif.latex?v'_i=\frac{v_i-min_A}{max_A-min_A}(newMax_A-newMin_A)&plus;newMin_A" />

### Z-score
Values are normalized based on the mean and standard deviation of A

<img src="https://latex.codecogs.com/gif.latex?v'_i=\frac{v_i-\bar{A}}{\sigma_A}" />

## Data Reduction

### Dimensionality reduction
Reduce the nubmer of random variables, or attributes under consideration. Valid approaches include wavelet transforms, principlae conomponent analysis, or attribute subest selection.

## Data Cube Aggregation

## Attribute Subset Selection

## Decision Tree Induction

### Numerosity Reduction
Replace existing representations with smaller, less-verbose options.

## Regression and Log-Linear Models

## Histograms

## Clustering

## Sampling

## Discretization

## Concept Hierarchy
- street < city < state < country
- {Cookeville, Knoxville, Nashville} < Tennessee


## Segmentation by Natural Partitioning

## 3-4-5 rule

## Automatic Concept Hierarchy Generation
- based on number of distinct values per attribute in dataset
- most distinct values is at the bottom of the hierarchy
- Example:

|Attribute | Distinct Values|
|:---------|-----------:|
|Country|15|
|province/state|365|
|city|3,567|
|street|674,339|

