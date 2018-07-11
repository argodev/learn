# Chapter 5: Compiling OpenACC

This chapter struck be a bit oddly both in its title and placement. However, the more I chew on it, the more I appreciate the material. Essentially, the author of this section aims to help the reader think about the work that a generic OpenACC compiler has to perform. To think through what aspects of this are hard for the computer, which are easy, where does it excel, where might it fail.

The chapter starts with what seems a primer to parallel computing and the associated challenges of parallelizing loops (true data independence). The author highlights a number of "gotcha" cases that may get missed on a less thoughtful review of the code. He spends a little time discussing how these challenges map to the directives provided by OpenACC and what the compilers generally do in various situations (fail safe).

The next section goes into detail about what the compilers are good and bad at and similarly what humans are good and bad at - essentially illustrating the co-dependence of the two in effectively parallized applications. 

The last major section in this chapter addresses how the compiler treats the directives added by the developer. Some discussion is given to whether the directives are _prescriptive_ or _descriptive_. In the first case, the compiler blindly follows the instructions assuming that the developer knows best. The latter scenario, the compiler understands the intent of the developer but it retains veto power. My reading of this section leads me to believe that the specification is not explicit on this topic and the individual compilers are free to address them as desired.


[<< Previous](../Chapter_04/readme.md)
|
[Next >>](../Chapter_06/readme.md)
