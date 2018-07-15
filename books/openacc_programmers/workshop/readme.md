# Workshop Outline

I have been thinking about how I would structure a workshop on OpenACC given both what I know and the contents of this book. 

One strong starting point is the [GitHub repo for this book](https://github.com/OpenACCUserGroup/openacc_concept_strategies_book). Here not only will you find the code samples for the various chapters, but you will also find slides and some simple overview recommendations for teaching through the book.

Probably the one main thing I would change would be to attempt to simplify the requirements and provided code samples prior to teaching. Meaning, some of the samples are difficult to run if you do not have the compiler that is preferred by the author of that particular chapter. Early in the book it appeared that the PGI Community Edition was preferred and I spent some time ensuring I had that installed/running on my system only to find in later chapters that I also needed to have the Intel (commercial) compilers installed. I'd recommend walking through the code samples and adjusting them to all utilize the PGI compiler (make files, etc.).

Additionally, where possible, I would recommend attempting to pick a single codebase that can be built upon/modifified as one works through this book. This would allow the reader to focus on the concepts germaine to that chapter and not need to be distracted by the uniqueness of a given code base.

[<< Previous](../Chapter_12/readme.md)
|
[Next >>](../readme.md)
