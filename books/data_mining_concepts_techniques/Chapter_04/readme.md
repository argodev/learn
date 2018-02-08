# Chapter 4: Data Warehousing and Online Analytical Processing

## What is a Data Warehouse?
There are a number of things (based on the text) that describe a data warehouse. These are some of the onest that stuck out to me:

- Physically separate store (segmented from the online transactional database)
- Subject-oriented: organized around major subjects rather than optimal structures for transactions
- Integrated: Usually pulls together data from various sources
- Time-variant: almost every record has some time component to it (historical data)
- Nonvolatile: Updates only occur in scheduled batches... as such, much of the normalized structure (transaction support, etc.) aren't needed.


Requires two operations: initial loading and access

Query Driven
- This is the microservices approach... I have a number of smart agents that go and get the data I need when I ask

Update Driven
- warehouse is populated on scheduled update rate
Warehouse is update-driven, high-performance

Information is periodically moved over, integrated and pushed to the warehouse

For frequent queries....

Update driven is better... effectively caches the integrated sources

OLTP: current, actionable

OLAP: historical, strategic

Mutli-tiered architecture

Data Mart – a snapshot of some part/sub-piece of the warehouse

Conceptual Model

Star

Fact table in the middle

Need to be able to draw a snowflake schema given a scenario

Snowflake

Kinda like a star, but multi-layered... facts go to others which go to others, etc.

Fact Constellations

Multiple fact tables

Circular references

Measures of Data Cube: Three Categories

Distributive

Count, sum, min, max (map/reduce works)

Algebraic

Avg(), min() std_deviation

Needs to have all of them to make a full answer

Can have components that are distributive

Holistic

Median(), mode(), rank()

Concept Hierarchy: Dimension (location)

All -> region -> country -> city -> office

Multidimensional data

Sales Volume as a function of product, month, and region

Dimensions: Product, location, time

Hierarchical summarization paths

Industry -> Category -> Product

Region -> Country -> City -> office

Year -> quarter -> month -> day

-> week ->

Sample Data Cube

Roll-up/Drill-up

Climb the hiearachy, dimensionality reduction

Cities to country

Drill-down

Expand an existing attribute

e.g. quarters to months (add more dimensions)

Slice (carve)

3d -> 2d... cut off something

e.g. just tive me all the data for Q1

Dice (project out)

End up with a 3-d something, based on a query

Location = X and Y

Time = Q1 or Q2

Category = A and B

Pivot

Take a slice and change the way we look at it.
