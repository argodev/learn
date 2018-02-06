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

Sample Values for following items:
```python
data = [30, 36, 47, 50, 52, 52, 56, 60, 63, 70, 70, 110]
```

### Measuring Central Tendency
This means to do exactly what it sounds like... calculate the middle point, or tendency of the data... where do most of the values sort of go?

__Mean__: Average

Sample: <img src="https://latex.codecogs.com/gif.latex?\bar{x}=\frac{1}{n}\sum_{i=1}^{n}{x_i}" /> or Full Population: <img src="https://latex.codecogs.com/gif.latex?\mu=\frac{\sum{x}}{N}" />

Using our data from above, the mean is `58`.

Alternatively, you can use the Trimmed mean (chops extreme values). This requires the provision of a weight assigned to each value (_where does this come from?_)

<img src="https://latex.codecogs.com/gif.latex?\bar{x}=\frac{\sum_{i=1}^{n}{w_ix_i}}{\sum_{i=1}^{n}{w_i}}" />

__Median__: middle value of set.
<img src="https://latex.codecogs.com/gif.latex?{median}=L_1\left(\frac{\frac{n}{2}-\left(\sum{f}&space;\right&space;)l}{f_{median}}&space;\right&space;)c" />

Using our data from above, the median is `54`.

__Mode__: Value (or values) that occurs most frequently in the data. Can end up with bimodal or trimodal data
<img src="https://latex.codecogs.com/gif.latex?{mean}-{mode}\approx3x({mean}-{median})" />

Using our data above, the mode is `52` and `70`. The data is `bimodal`.

__MidRange__: Average of the largest and smallest values in the set
<img src="https://latex.codecogs.com/gif.latex?{midrange}=\frac{min()&plus;max()}{2}" />

The midrange of our data sample is `70`.

### Symmetric vs. Skewed Data
If plotted, where is the "hump"?
- Symmetric: hump is in the center. Often called "normal" distribution
- Negatively skewed: hump is on the left side. Mode is less than both median and mean
- Postively skewed: hump is on the right. Mode is greater than both median and mean

```python
import matplotlib.pyplot as plt
import seaborn as sns

data = [30, 36, 47, 50, 52, 52, 56, 60, 63, 70, 70, 110]
sns.set(color_codes=True)
sns.distplot(data)
plt.show()
```
![Distribution](distribution.png "Distribution")

### Measuring the Dispersion of Data
- Quartiles: Q1 (25th percentile - `47`), Q3 (75th percentile - `63`)
- Interquartile range: IQR = Q3-Q1 or `16`
- Five number summary: min, Q1, Median, Q3, max (note that this is *not* the mean). Our values are `30, 47, 54, 63, 110`
- Boxplot: visible version of the five-number summary. Median is marked, box start at Q1 and ends at Q3. Whiskers are at min/max unless there are outliers (> 1.5xIQR). If there exists outliers, the Whiskers mark the min/max values that are *not* outliers and then the outliers are marked individually.

> Note: For this data set, the IQR is `16`. Therefore any value more than 24 away from the median is considered an outlier (range is `30 - 78`).

```python
import matplotlib.pyplot as plt
import seaborn as sns

data = [30, 36, 47, 50, 52, 52, 56, 60, 63, 70, 70, 110]
sns.set(color_codes=True)
sns.boxplot(data=data)
plt.show()
```
![Box Plot](boxplot.png "Box Plot")




Variance (where sample: s, popuation: theta)
<img src="https://latex.codecogs.com/gif.latex?s^2=\frac{1}{n-1}\sum_{i=1}^{n}{(x_i-\bar{x})^2}=\frac{1}{n-1}\left(\sum_{i=1}^{n}{x_i^2}-\frac{1}{n}\left(\sum_{i=1}^{n}{x_i}&space;\right)^2\right)" />

Standard Deviation is the square root of the variance squared


<img src="" />
<img src="" />
<img src="" />


