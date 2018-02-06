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

## Data Integration

## Data Transformation

## Normalization

### Min-Max

### Z-score

## Data Reduction


## Data Cube Aggregation

## Attribute Subset Selection

## Decision Tree Induction

## Numerosity Reduction

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

