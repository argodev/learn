# Time Series Databases
_New Ways to Store and Access Data_
Ted Dunning and Ellen Friedman

## Chapter 1: Time Series Data: Why Collect It?

* one key is, what is the predominate access pattern? Is it time-based?

* time as an interval, an ordering sequence, or an absolute reference

* when it occurred may be different than when you learned about it

* typicaly (though not always) numbers

* often write-once, read many

* the notional story of Matthew Fontaine Maury in the mid-19th century... data-driven routing of ships, saved over a month on a trip from Baltimore, MD to Rio de Janeiro and back

* this was also an early example of crowd-sourcing... Maury "charged" captians with providing detailed records of their own travels in order to use his data. Early example of incentivization

* time series can allow you to see trends over time that would be invisible with aperiodic point measurements

## Chapter 2: A New World for Time Series Databases

* The increase in quantity of reporting devices is a source of pressure on tools

* Use a non-relational TSDB when you: 
  * have a huge amount of data
  * mostly want to query based on time
* They focus on hadoop-based TSDBs 

* Common TS Questions:
  * What are the short- and long-term trends for some measurement (prognostication)
  * How do several measurements correlate over a period of time (introspection)
  * How do i build an ML model based on the temporal behavior of many measurements correlated to externally known facts (prediction)
  * Have similar patterns of measurements preceded similar events? (introspection)
  * What measurements might indicate the cause of some event such as a failure (diagnosis)
