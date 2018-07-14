# Chapter 11: Innovative Research Ideas Using OpenACC, Part 1

This chapter (and presumably the next) are particularly intersting as they look forward to the future of OpenACC and ways in which it is or may be extended. This chapter is divided into two parts. The first deals with the Chineese-based Sunway Taihulight super computer and how OpenACC is supported there. The second half is written by a developer at NVIDIA who discusses another branch of OpenACC called OpenUH.

## Sunway OpenACC

This section was written by Lin Gan who works with the Sunway super computer on a regular basis. Prior to talking about how they extended the OpenACC platform (they call their variant OpenACC*), he starts by giving a description of the machine and explaining how it works. This is critical as it utilizes a custom chip which has, as you might imagine, unique computational properites and does not exactly map to the OpenACC standard. I did find it intersting, however, to see how they designed the chip and the tradeoffs that they made in their push for super computer dominance.

Due to the uniquness of their architecture, they were forced to develop their own compiler and, in the process, added some custom directives. Specifically, they handle some of the data movement directives differently than one might normally expect (some of them are no longer needed as the device and host can share aspects of memory).

## OpenUH

This section dives deep into the arena of loop scheduling algorithms. Specifically, they are looking at what the compiler does when you insert a directive indicating that something should be parallelzied. This work has led them to propose a few extensions to the platform (candidate clauses). They then go on to implement some of their various scheduling algorithims and test them on live systems. The results for various scenarios are presented at the end of the chapter.

As I read this last section, I found myself wondering how it aligned with the advice given earlier in the book on portable code. In many ways, these sorts of optimizations seem very system-specific and, while they may help you eek out that last bit of performance on one system, it may have widely different implications on another.

[<< Previous](../Chapter_10/readme.md)
|
[Next >>](../Chapter_12/readme.md)
