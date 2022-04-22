# Chapter 2 - Device Architectures

> NOTE: There are no code examples/exercises for this chapter.

- wide range of industry groups worked together
- primatives for detecting available hardware
- in spite of goals, platform independence is not yet obtained
- makes a point that devs must still understand the capabilities of particular hardware devices... not sure that is ever going to go away
- programming effort is a key consideration in "performance" equation
- they discuss briefly that you can optimize differently at the same time... you can have various processors running in the same system at the same time with different algorithm optimizations based on the characteristics of a given code and targeted processor

## Topics discussed in this chapter

- SuperScalar execution
- Very Long Instructin Work (VLIW)
- SIMD and Vector Processing
- Hardware Multithreading
- Multicore Architectures
- Systems-on-Chip and APUs
- CPU Architectures
- GPU Architectures

## Items of note (to me at least)

This chapter was quite helpful as it discussed the many different approaches to parallelisim and the various chips that support them. Prior to this point, I had been quite aware of logical parallelisim - essentially enforced or enabled by the developer in code based on his/her understanding of the inter-dependencies (or lack of them). I had in my head, a spectrum that went from tightly coupled (high communicate/compute ratio) to fully data parallel (low/no communicate/compute ratio). While this is not an incorrect understanding, I did not appreciate the work by compilers (VLIW) and chip designers with regards to developer-free parallelism. I appreciated the the path the authors took through the material as I felt it built reasonably well on top of itself such that the CPU/GPU/APU content at the end was easy to understand based on the prior topics.

- [out-of-order logic](https://en.wikipedia.org/wiki/Out-of-order_execution): my understanding of this topic is that it is a feature of some CPUs that will assess the way in which a programs' intructions will be utilizing the processor and re-order the instructions (where possible) to more efficiently use the chip. It maintains a map that allows the code to reassemble the results of the computation on the other side.
- [Very Long Instruction Word (VLIW)](https://en.wikipedia.org/wiki/Very_long_instruction_word): this is a compiler feature that looks at the instructions and attempts to "bundle them together" into units that it thinks will be run together. At run time, the processor is presented with a longer instruction that (in theory) utilizes more of the processor at an instant in time. This is reminiscent of the approach that hard drives take (reading the data around what you asked for on the assumption that the liklihood is high that you will ask for it as well).

[<< Previous](../Chapter_01/README.md)
|
[Next >>](../Chapter_03/README.md)