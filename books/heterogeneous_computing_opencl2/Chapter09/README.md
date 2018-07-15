# Chapter 9 - Case Study: Image Clustering
This chapter aims to instruct by tackling a real-world problem and walking through a possible solution/implementation. Specifically, they tackle the histogram portion of the feature generation task supporting bag-of-words classification for images.

This is another one of those chapters that should live on as a reference for future projects. The small nuggets of wisdom discussed in each refactoring of the code are pretty significant as well as nuianced.

They start out with a "naive" GPU implementation (their words). This serves as the baseline for subsequent comparisons. The second try introduces coalesced memory accesses (grabbing more than you need, fewer times, resulting in fewer memory accesses). The third adds vectorization of the computation. The fourth approach moves the features into local memory in an attempt to benefit from the high-speed cache in some GPUs. The fifth implementation is similar to the fourth, but rather than using local memory (work-group specific), it stores some of the data in constant memory (again, leveraging cache improvements). Once the code listings for each implementation is provided, they perform a rudimentary performance analysis using different size/complexities and testing each of the 5 impelementations.

# Results
As with many things, the results are not always completely clear as the benefit (or not) of a given optimization will vary with data size or other input parameters. In all cases, option 2 (memory coallescing) performed significantly better than option 1. In some cases, by over an order of magnitude. Option 5 was the most consistently-good option though option 3 (vectorized compute) was a close second. Option 4 (local memory) often performed worse than 3 or 5. 

# Take Away
The biggest thing this chapter highlights (in my mind) is that "getting it to work" is not the end-goal for GPU-targeted algorithms. Rather, that should be accomplished in the first 10% of the work effort or so, and the remainder of the time focused on ratetching up the performance. The difference between the first version and the last was significant with option 5 taking around 2% of the time required for option 1. That is too significant to ignore.
