# Chapter 12: Innovative Research Ideas Using OpenACC, Part 2

This last chapter is similar to the prior in that it is looking forward to new work that is occuring in the OpenACC space. The first portion is written by a colleage of mine, Seyong Lee of ORNL and is focused on using OpenACC in the context of reconfigurable devices such as FPGAs. The second half is written by Jinpil Lee from Japan and discusses how they have modified the Omni compiler for OpenACC.

## A Framework for Directive-Based High-Performance Reconfigurable Computing

Seyong and his group have extended the research-focused OpenARC compiler (an OpenACC-aware compiler) such that it will generate the appropriate code for FPGAs much the same as it would for GPUs. This is notionally similar to what OpenCL does in its more recent versions - something Seyong alludes to - but aims to do so at a higher-level that is more consumeable by more developers. So closely related are these to efforts that, at present, this project generates OpenCL code (rather than raw VHDL or Verilog) as an interim step. This is important as one of the weaknesses of FPGAs is the high knowlege barrier required to effectively program them at scale. 

In their work, they discovered a handful of areas wherein additional directives and options in the OpenACC space may help a compiler better optimize for FPGAs. I've briefly listed them below:

- Kernel Configuration Boundary Check Elimination
- Directive Extension for Loop Unrolliung, Kedrnel Vecvtorization, and Compute Unit Replication
- pipe, pipein, pipeout (alternative data clauses)

This section ends much the same as an academic paper... with emperical evidence supporting their proposed changes. They ported 8 benchmark tools and then ran them on four different devices: FPGA, Xeon Phi, NVIDIA Tesla, AMD Radeon.

The results are much along the lines of what you would expect. For certain applications the GPUs perform better and others the FPGA does. In nearly all cases, the accelerators perform better than the CPUs. They do assert that GPU-based accelerators perform better at wide, massively parallel operations wereas the FPGA blew them away when looking at deeply pipelined codes.

## Programming Accelerated Clusters Using XcalableACC

This section starts with an introduction to XMP - their implementation language for multi-node development. It appears (from the outside, at least) that they have internally standardized on this and found it easier/clearer than MPI for message passing problems. 

They then proceed to discuss XACC - their extension of XMP with OpenACC-like directives. To a large degree, it appears that they have simply borrowed much of the spec and view it as a way to help them leveate the inter-node parallelism that is availalbe (with XMP handling the node-to-node level).

They end their section by discussing a series of experiments they performed wherin they compared XACC to MPI+OpenACC. In some cases, XACC performed more poorly, but they explained this as being an issue with the wa they had chosen to implement some MPI-style barriers. It appears that, at least for their platform, XACC is a viable and interesting option.

[<< Previous](../Chapter_11/readme.md)
|
[Next >>](../workshop/readme.md)
