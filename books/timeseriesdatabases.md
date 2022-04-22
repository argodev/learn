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


## Chapter 3: Storing and Processing TIme Series Data

* Legitimate questions must be addressed
  * how many different time series are there?
  * what kind of data is being acquired?
  * at what rate is the data being acquired?
  * for how long must the data be kept?
  * for how long must the data be kept at the same resolution (e.g is rolling-up allowed)?
* Flat Files
  * Parquet file format?
  * note that flexibility in file formats might lead to increased verbosity/storage costs
* Star Schema (RDBMS)
  * Can work reasonably well up to hundreds of millions or billions of data points
  * storing is one thing, retrieving and processing it is another
* NoSQL with Wide Tables
  * problem is in storing one data point per row - decreases the amount of data returned per read
  * wide tables (single metric, each column is a timestamp)
  * has a series id, time-window start time, and then each column is an offset of some value
  * usually has between 100-1,000 samples per row
* NoSQL with Hybrid Design
  * has both uncompressed and compressed data in the same row.
  * chron-like job runs and compresses row contents on some schedule
* Direct Blob INsertion Design
  * use memory like a cache to hold data until it is ready to be compressed into blobs and written to the storage layer
  *``



