---
title: Comprehensive Exam - Question \#1
author: Rob Gillen
header-includes:
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[RO,RE]{Comprehensive Exam - Question \#1}
    - \fancyhead[LO,LE]{Rob Gillen, T00215814}
    - \usepackage{tikz}
    - \usetikzlibrary{calc,shapes.multipart,chains,arrows}
bibliography: references.bib
---

# Question

Select one of these 2 papers (_Anomaly detection in cyber physical systems using recurrent neural networks_[-@7911887] or _Checking is believing: Event-aware program anomaly detection in cyber-physical systems_[-@DBLP:journals/corr/abs-1805-00074]), and critique it and the work described in it. Then describe how the potential methods and measurements you want to investigate would aide in evaluating the sceptibility of their proposed approach. Then, in one of the real-world domains you want to explore, discuss how the author’s proposed approach compares and contrasts to what you are proposing to do.

# Answer

## Anomaly detection in cyber physical systems using recurrent neural networks

In this paper, the authors present an approach to detecting anomalous traffic in a cyber-physical system.  

- unsupervised
- RNN as a time-series predictor
- cumulative sum method to identify anomalies
- "majority" of attacks
- "low" false-positive rate
- Utilize Long Term Short Term Memory RNN (LSTM-RNN) to correlate time series data

Contributions

- Modelling normal behavior in CPS using an unsuperivsed, deep learning approach
- identify the sensors that exhibit the anomalous behavior
- validation of the approach on the Secure Water Treatement Testbed (SWaT)

Novelty

- Work is in water critical infrastructure, specifically SWaT
- This environment reflects the complexity normally found in a real-world plant
- approach uses time-based neural networks to consider sequence of information
- not only detects anomalies, but also the source (sensor in question)

- they describe what LSTM does and how it works... it doesn't appear that they built/did it themselves (just implemented it)



### Review

asdf

### how would my work measure/evaluate the suseptibility of their approach

### Using a real-world domain I'm going to explore, how does the author's proposed approach compare/contrast to my work

## Checking is believing: Event-aware program anomaly detection in cyber-physical systems

### Review

asdf

## References