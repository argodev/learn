---
title: Comprehensive Exam - Question \#3
author: Rob Gillen
header-includes:
    - \usepackage{fancyhdr}
    - \pagestyle{fancy}
    - \fancyhead[RO,RE]{Comprehensive Exam - Question \#3}
    - \fancyhead[LO,LE]{Rob Gillen, T00215814}
    - \usepackage{tikz}
    - \usetikzlibrary{calc,shapes.multipart,chains,arrows}
bibliography: references.bib
---

# Question

How may evolving computer architectures affect and be affected by future cyber-physical systems?  In general consider from low level architecture such as processor, cores, gpu, etc.  to higher levels such as cloud, edge, fog computing, etc.  and embedded systems like those going into vehicles and sensors.  Please limit your response to at most 3-pages as I realize this alone could make a dissertation level discussion.

# Answer

- Discuss embedded processors such as the Intel Movidius VPU and neural compute stick[-@movidius]

- Also the Xilinx Zync UltraScale+ MPSoC[-@zynq]

- Also the rack-mount computers provided by Schweitzer Engineering Laboratores[-@sel_computers]

- Self-driving cars hardware such as the NVIDIA Drive AGX Pegasus and AGX Xavier[-@nvidia_agx]. 320 TOPS (Tensor Operations per Second) or 39 TOPs six different types of processor for redundant and diverse deep learing algorithms

- Advanced processors may be used for core operations but not as likely (exceptions will be vehicles, etc.)

- Advanced processors may likely be used for monitoring and security of said networks (Deep-learning based inferences, etc.)

- GPU-based processors deployed in onboard systems for edge-based image processing and data reduction in UAS and remote sensing.

- Darpa project[-@darpa_hackfest] utilizing Ettus E310/312[-@ettus] onboard for custom RF communications, custom sensing, data processing, etc.

- talk about how the confiuration of many electrical grid/SCADA systems mimic the fog/cloud relationship and the general notion of sensor aggregation frameworks.

- work on data aggregation in sensor networks has been evolving for years[-@5693294]

- Potok and Schuman work on combining Quantum, HPC and Spiky Neural Networks for low-power inferencing[-@potok2016study]

- Work Schuman and team from UTK are doing with NeoN: Neuromorphic Control for Autonomous Robotic Navigation[-@mbd:17:neon]. They utilized Titan to train and optimize the network that then ran on their FPGA-based implementation of the neual network.

- GPGPU enabled platforms for C4ISR mission space by Curtiss-Wright[-@curtiss:wright]

## References