# Chapter 11 - Mapping High-Level Programming Languages to OpenCL 2.0

This chapter diverges a bit in that rather than focusing on OpenCL specifically, it introduces C++ AMP which is a high(er)-level language that is focused on developer productivity and allowing developers to expressively convey parallel constructs (e.g. parallel.for() from C#).

They then discuss what it would take to build an intermediary compiler that could take a developer's inent as described in C++ AMP and generate the appropriate OpenCL code. They actually go so far as to do so and run performance tests comparing both approaches.

Interestigly, the generated kernel runs in almost identical time to the hand-coded kernel. However, the ceremony in C++ required to automatically generate/compile the kernel takes significantly longer (runtime-wise) than the hand-crafted version (no surprise). As with most of the rest of this topic, the tradespace needs to be fully considered between developer productivty/accuracy and runtime performance.


