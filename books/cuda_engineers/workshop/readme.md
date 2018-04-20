# Workshop Outline

One objective of working through this text was to plan out a workshop on GPU programming and to assemble some thoughts as to the material and adgenda that would be utilized. This page includes that general structure.

https://developer.nvidia.com/cuda-education

## Day 1

### Introduction

- Why would we want to use GPUs?
- How do they differ from CPUs? 
- What other processor types are there? (ASIC, FPGA, DSP, etc.)
- What caused the rise in popularity?
- What are they good for?
- What are they bad for?
- Other methodologies (OpenCL? OpenACC, etc.)
- CUDA Specifics?

### Setup/Testing 

- Cover appendicies here
- Get machines up and running
- Walk through building, testing, seeing, debugging an app
- This is a long section, but gets everyone working on the same page

### Overview of GPU Stuff

- Dig deeper into the hardware and how it affects the processing (tri-mode grids)
- Do we need this section?
- Might cover some of the Chapter 2 overview details (execution context, syntax, terms, etc.) 
- Use as opportunity to clarify some of the things seen in the setup/testing section.
- Walk through Chapter 3 quickly... distv2, the standard workflow section, cudaMallocManaged, etc.

### 2D Real-World Apps

- This is where things start to get interesting. Take this section and do a deep-walk through on Chapter 4... There's alot to consume and alot to appreciate.
- Skip the dist_2d section (besides maybe discussion) and move to dist_rgba. 
- Once that is finished, walk carefully through the flashlight app example and ensure everyone can make it work. Let them play with it a bit and see how fast/smoothly it renders on their machines
- The next section walks through the generalization of the flashlight app... the stability app... walk through the similarities and differences and get it working.
- Discuss how this might map to real-world applications

## Day 2

### Deep dive on Shared Memory
This is all about chapter 5
neighboring cells, etc.

### Reduction/Atomics
Similar to the above 

- [Chapter 5: Stencils and Shared Memory](Chapter_05/readme.md)
- [Chapter 6: Reduction and Atomic Functions](Chapter_06/readme.md)
- [Chapter 7: Interacting with 3D Data](Chapter_07/readme.md)
- [Chapter 8: Using CUDA Libraries](Chapter_08/readme.md)
- [Chapter 9: Exploring the CUDA Ecosystem](Chapter_09/readme.md)