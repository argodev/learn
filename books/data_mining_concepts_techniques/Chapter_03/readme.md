# Chapter 3: Data Preprocessing

## Major Tasks
- Cleaning: Fill in missing vlaues, smooth noise, identify/remove outliers, address inconsistencies
- Integration: combine multiple sources
- Transformation: normalization/aggregation
- Reduction: reduce representation with same/similar results
- Discretization

### Some of Many ways to deal with missing data
- Ignore
- Fill in the missing value manually
- Use a global constant to fill in the value
- Use the mean or median
- use the mean or median of the class of records the data otherwise fits in
- use the most likely value

## Binning Methods for Data Smoothing
Once your data has been binned (via equal-width or equal-depth), you may choose to smooth the data. Common approaches include mean (all values in the bin are replaced with the average) or boundary (first half of the values get the min boundary value, second half get the upper boundary value)

Note, similar approaches can be used for data reduction... in this case, N values may be replaced with a single that is the mean, etc.

## Data Integration
This involves the merging of data from various sources. While the idea sounds simple, there are a number of issues that may arise:

- Entity Identification issues (bill vs william, cust_id vs. cust_number etc.)
- Redundancy
- value conflict and resolution (which one wins?)


## Data Reduction
Goal is to produce a reduced (in size) set of data that closely maintains the integrity of the original data. Results should be the same (or nearly so) as the original data.

__Dimensionality reduction__: Reduce the nubmer of random variables, or attributes under consideration. Valid approaches include wavelet transforms, principlae conomponent analysis, or attribute subest selection.

__Numerosity Reduction__: Replace existing representations with smaller, less-verbose options.

__Principal Component Analysis__: Algorithimic way to decide which attributes actually matter.

__Attribute Subset Selection__: Similar to PCA, but done more analytically... which values can be tossed? Which do not contribute in any meaningful way to the solution?

__Decision Tree Induction__: While normally used for classifcation, this can be used as a means of attribute reduction.Buid a tree, use the "Best" attribute at each level to divide and then, when finished, eliminate all attributes not used in the decision making process.

__Regression and Log-Linear Models__: Use probability distribution functions and the ability to project from fewer-dimensional space into higher dimensional space to then use the lower-dimensionality to represent the data

__Histograms__: See the binning reduction approaches described above (Equal-Width, Equal-Frequency). These are valid means for reducing the number of data points to be considered

__Clustering__: Build a set of clusters with a minimum centroid distance and then use that attribute rather than each individual value


## Sampling
This is subsetting the data by pulling some smaller portion of it out. A couple of approaches are valid:

- Simple Random without replacement: pull a value randomly. Once it is pulled, it cannot be pulled again

- simple Random with replacement: pull a value randomly. Once pulled, it *can* be pulled again.

- clustered samples: Cluster the records and then pull equal samples from each cluster

- Stratified sample: break the data into groups based on some attribute and then sample them randomly based on the same distribution as in the original data

## Data Cube Aggregation
As described more in chapter four, use data cubes to aggregate the data for subsequent analysis. Generally this tosses the transactional records in favor of summarized rows/data.



## Data Transformation

__Smoothing__: Works to remove noise from the data. Techniques include binning, regression, and clustering

__Attribute Construction__: Also called Feature Construction - new attributes are constructed based on existing ones. Sometimes referred to as augmentation

__Aggregation__: Sums (or similar operations) of the data into monthly or annual amounts. Flattening many records into fewer ones for analytical purposes.

__Normalization__: This is when we scale attribute data such that is falls into a smaller range, such as -1.0 to 1.0, or 0.0 to 1.0.

- _Min-Max_: Linear Transformation on the original data

<img src="https://latex.codecogs.com/gif.latex?v'_i=\frac{v_i-min_A}{max_A-min_A}(newMax_A-newMin_A)&plus;newMin_A" />

- _Z-score_: Values are normalized based on the mean and standard deviation of A

<img src="https://latex.codecogs.com/gif.latex?v'_i=\frac{v_i-\bar{A}}{\sigma_A}" />

__Discretization__: raw values of numeric value are replaced with interval labels (0-10, 11-20) or conceptual labels (youth, adult, senior). These then can be used to help form a concept hierarchy.

- Equal-width partitioning
- Equal Depth (frequency)


## Concept Hierarchy
- street < city < state < country
- {Cookeville, Knoxville, Nashville} < Tennessee

## Segmentation by Natural Partitioning
- If an interval covers 3, 6, 7, or 9 distinct values at the most significant digit, partition the range into 3 equi-width intervals
- if it covers 2, 4 or 8 distinct values at the most significant digit, partition the data into 4 intervals
- if it covers 1, 5, or 10 distinct values at the MSD, partition into 5 intervals


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

