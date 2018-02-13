# Chapter 6: Mining Frequent Patterns, Associations, and Correlations: Basic Concepts and Methods

Discussion of the paper from last time... Business Information Technology
- data compression (reducing storage, memory sizes, network transfer sizes, etc.)
- map/reduce
- loading tools
- 3rd party products, cloud services, etc.

## What is Frequent Pattern Analysis
- Patterns that occur frequently (no surprise)
    - set of items
    - subsequences
- Motivation: Find the inherent regularities in the data
    - if you do X, what is the liklihood you will do Y?
- Applications
    - basket data analysis, cross-marketing, catalog design, sale campaign analysis, web log /click stream, DNA sequence

## Why is it important?
- Forms foundation for many other algorithms:
    - Association, correlation, causality
    - Sequential, structural patterns
    - Pattern analysis in spatiotemporal, multimedia, time-series, stream
    - Classificaiton: associative classification
    - Cluster analysis: cluster by pattern
    - Data warehousing: iceberg cube and cube-gradient
    - Semantic data compression: fascicles
    - Broad applications

## Basic Concepts
- Itemset $X = {x_1, ..., x_k}$
- Find all rules $X --> Y$ with minimum support and confidence
  - support, s, probability that a transaction contains $X \union Y$
  - confidence, c, conditional probability that a transaction having `X` also contains `Y`

Considered interesting if they meet some minimum support and minimum confidence thresholds

## Computational Complexity of Frequent Itemset Mining
- How many itemsets are generated in the worst case? M^N (M: # of distinct itmes, N: max length of transaction)
- Wost case complexity vs. expected probability

## Apriori: A Candidate Generation and test approach
- If there are any itemset that is infrequent, it's superset should not be generated/tested
- Method
  - scan db 1x to get frequent 1-itemset
  - generate length k+1 candidate itemsets from length k frequent itemsets
  - test the candidates against the db
  - terminate when no frequent or candidate set can be generated
- Example/walkthrough

## Challenges
- Challenges
    - Multiple scans of transaction db
    - huge number of candidates
    - tedious workload of support counting for candidates
- Improving: general ideas
    - reduce passes of transaction db scans
    - shrink number of candidates
    - facilitate support counting of candidates

## DHP: Reduce the number of candidates
- A k-itemset whose corresponding hashing bucket count is blow the threshold cannot be frequent
- J. Park, M. Chen, and P. Yu: An effective hash-based algorithm for mining

## Partition: Scan database only twice
- Any itemselt that is potentially frequent in DB must be frequent in at least one of the partitions of the DB
    - scan 1: partition database and find local frequent patterns
    - scan 2: consolidate global frequent patterns

## Sampling for Frequent Patterns
- Select a sample of the original DB, mine the subset for patterns
- Scan database once to verify frequent itemsets found in sample, only _borders_ of clsure of frequent patterns are checked
  - e.g. check _abcd_ instead of _ab_, _ac_, ... etc.
- Scan database again to find missed frequent patterns

## Bottleneck of Pattern Mining
- Multiple db scans are __costly__
- mining long patterns needs many passes of scanning and generates lots of candidates
- Bottleneck: candidate-generation-and-test
- Can we avoid candiate generation?

## Avoid candidate generation
- grow long patterns from short ones
- FP-Tree/FP-Growth Example
- Example/walk-through of FP Growth
- Example/walk-through of FP Growth and APriori
