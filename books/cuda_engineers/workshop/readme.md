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

- Spend time talking about thread interactions, shared memory, etc.
- work through code examples (heat and sharpening)
- spend *some* time with the perf tool and the sharpening code, but don't let it get carried away... it would be easy to spend too much time here

### Reduction/Atomics

- Briefly explain the role of atomics/reductions (e.g. map reduce lite)
- focus on the second example (map/image) and try to ensure that they are able to get it working as well as understand what happened.

### 3D Data

- This session could be skipped if needed
- The example is interesting but feels a bit of a further stretch for most and I'd rather focus on the Chapter 8 content if limited

### CUDA Libraries

- Plan to spend most of the afternoon here
- Walk through the libraries referenced in Chapter 8, talk about how to use them
- work through the examples (manually reconstruct... don't just copy/paste)
- While the background to this point has been helpful, this section is likely more immediately useful, so settle in and relax

### Where to next?

- Spend the last little bit of the day discussing a path forward
- What other tools/options are available (OpenACC? OpenCL? AMP+, etc.)
- What libraries/tools are available?
- Collect feedback from the workshop and dismiss

[<< Previous](../Appendix_D/readme.md)
|
[Next >>](../readme.md)
