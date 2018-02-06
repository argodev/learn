# Chapter 2: Getting to Know Your Data

## General Types of Data Sets
- Records: traditional database table/excel sheet mental model, term frequency vector, etc.
- Graphs: network or social information structures
- Ordered: Sort of like a record, but the field may contain multiple items

## Characteristics to consider
- Dimensionality
- Sparsity
- Resolution
- Similarity

### Attribute Value Types
- Nominal: profession, id numbers, eye color, zip codes
- Ordinal: rankings (relative to each other), military, grades, etc.
- Binary: positive/negative (has cancer/not)
- Interval: calendar dates
- Ratio: length, time, counts
- Discrete: finite (or countably infinite) set of values
- Continuous: real numbers, infinite precision is possible

## Ways to Understand your Data

### Measuring Central Tendency
This means to do exactly what it sounds like... calculate the middle point, or tendency of the data... where do most of the values sort of go?

__Mean__

Sample: <img src="https://latex.codecogs.com/gif.latex?\bar{x}=\frac{1}{n}\sum_{x=1}^{n}{x_i}" />

or

Full Population: <img src="https://latex.codecogs.com/gif.latex?\mu=\frac{\sum{x}}{N}" />

Alternatively, you can use the Trimmed mean (chops extreme values). This requires the provision of a weight assigned to each value (_where does this come from?_)

<img src="https://latex.codecogs.com/gif.latex?\bar{x}=\frac{\sum_{i=1}^{n}{w_ix_i}}{\sum_{i=1}^{n}{w_i}}" />




<img src="" />
<img src="" />
<img src="" />



