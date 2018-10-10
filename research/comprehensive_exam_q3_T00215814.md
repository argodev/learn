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

It would be fair, at almost any point in computational history since the mid-1980's to comment on the rapid evolution of the industry over the prior five-year period. While the past half-decade has not seen a dramatic increase in the clock speed of available processors, it *has* seen the emergence of, or perhaps the *return to*, computational-load-specific processors. Contrary, however, to many of the designs of early computers, these heterogeneous processors are regularly implemented on the same piece of silicon as other processor types - yielding benefits ranging from performance increases due to optimized interconnects to reduced package size and power requirements. The following sections of this paper discuss the potential impact of these advances in critical infrastructure, advanced sensors, and the data movement and processing of those systems.

## Core Infrastructure

One broad category of cyber-physical systems (CPS) is that of the control systems supporting the critical infrastructure. This includes devices automating the management and protection of our electric grid, water treatment plants, natural gas and oil pipelines and other similar systems. While this category of CPS will certainly benefit from advanced processor architectures, it will likely be the slowest to do so due to extreme size of the current install base, low profit margins and regulatory hurdles. Many infrastructure devices are running bare-metal applications on standard integrated circuits (ICs) or customized Linux on low-powered embedded processors.

Exceptions to this assertion can be seen in examples from Schweitzer Engineering Laboratories (SEL) and Sandia National Laboratories(SNL). In the case of SEL, they are developing hardened compute platforms for industrial deployment capable of expansion with GPUs and FPGAs[-@sel_computers]. The research at SNL has resulted in the Weasel Board[-@weasel_board] - an advanced processing and machine-learning based monitoring system for PLC devices. What makes this particular platform interesting is that it is designed to fit directly in the chassis backplane (retrofit rather than replace) reducing the fiscal resistance to deployment.

## Advanced Sensors

Another broad category of cyber-physical systems is advanced sensors. Used in many applications, two exemplars are autonomous systems and monitoring/protection networks.

### Autonomous Systems

Autonomous systems can simultaneously be viewed both as self-contained as well as fully connected cyber-physical systems. Examples of devices in this category include "smart" or "driverless" cars, both manned and unmanned aerial systems (UAS), and robotics. The evolution of heterogeneous computing systems have already broadly impacted these systems. For example, NVIDIA - originally known for its graphics cards that make games more realistic - are supporting the automotive industry with their Drive AGX Pegasus and AGX Xavier[-@nvidia_agx] platforms. These platforms provide 320 or 30 TOPS (Tensor Operations/Second) respectively - both providing a significant amount of computing power to support image detection, deep learning, and related sensing capabilities. These devices represent a massive leap forward when compared to the ECUs (Electronic Control Units) included in most modern vehicles.

Aerial systems such as military planes and helicopters as well as unmanned systems are also benefiting from advances in computing architectures. Military vendors such as Curtiss-Wright are providing GPGPU-enabled computational platforms for advanced on-board processing designed for C4ISR (Command, Control, Communications, Computers, Intelligence, Surveillance, and Reconnaissance) applications[-@curtiss:wright]. On the other end of the spectrum, DARPA recently hosted a "hackfest"[-@darpa_hackfest] wherein they invited participants to develop unique sensing and coordination algorithms for small UAS platforms that had been equipped with the Ettus E310 software-defined radio[-@ettus] which is powered by a Xilinx Zync MPSoC (combining both FPGAs and multiple ARM cores).

Recent work[-@potok2016study] in computational architectures shows the interplay between Quantum computing, High Performance Computing (HPC) and neuromorphic computing utilizing high-power and advanced computing to optimize models to be run on extremely low power devices (inferencing). This work has materialized in navigation systems supporting autonomous navigation for robotics[-@mbd:17:neon] wherein each decision consumes less than 1 picojoule.

### Protection Networks

One area wherein advances in processor architectures will most easily integrate into cyber-physical systems is that of monitoring and protection. protection systems are often "bolt on" or can be thought of as logically disconnected from the main operations of the system. Passive network taps allow these systems to watch and report on what is occurring in the environment without requiring significant changes to the extant equipment. Further, many recent protection algorithms are based on machine learning or deep learning techniques that either require or at least significantly benefit from alternate processor architectures. Devices such as the neural compute stick[-@movidius] based on the Intel Movidius VPU provide low-power options for the inferencing operations of deep learning networks on embedded systems. More customized solutions such as the Xilinx Zync UltraScale+ MPSoC[-@zynq] provide multiple general purpose ARM cores, multiple Real-Time ARM cores, FPGA fabric, GPU, and multiple communication options (CAN, Ethernet, SPI, etc.) all on a single chip. This breadth of processing types enable advanced analytic approaches (anomaly detection, traffic classification, neural networks, etc.).

## Data Movement/Processing

Research in data aggregation, summation, reduction and optimization from sensor networks and control networks has been advancing for many years[-@5693294]. The recent popularization of terms such as "fog", "cloud", and "edge" computing have, to a large degree, simply provided a vocabulary for activities that have existing for decades. SCADA networks often consist of RTUs (Remote Terminal Units) that serve not only as a communications bridge between higher-level compute systems and the physical sensors/controllers but also function as data aggregation/summation nodes (e.g. small-scale "fog"). The RTUs then forward a subset of the data on to the central control center that may contain large-scale compute assets ("cloud") where advanced processing can occur and results be made available to decision makers.

What *has* changed in recent years is not necessarily the overall theory (tree structure, aggregate/analyze at the lowest level possible, only centralize the data that must be), but the *means* and *methods* available to accomplish these objectives. Cloud Computing, for example is not a new concept, but rather has *democratized* and *broadened the availability* of computing. Similarly, the advances in computational architectures are enabling tremendous advances in the complexity of the compute operations that can occur in a given power envelope. This permits more work to happen at the edge (end point) which both reduces the amount of data that must transit limited communication links while increasing the end-device capability (edge). Examples of this provided previously in this paper include autonomous systems such as self-driving cars, C4ISR platforms and UASs.

In the context of UAS platforms the DARPA hackfast mentioned previously illustrates that advances in computational architectures permit the notion of independent swarming (e.g. without the coordination of a ground station). This capability may allow the platforms to accomplish an objective that would not be possible previously due to congested communications links and physics-based latency.

C4ISR platforms that can perform complex onboard processing via devices such as those provided by Curtiss-Wright can support sensors with increased fidelity - often far greater than the comms link could support. In the extreme case, advanced computing architectures allow not only airborne/deployed "edge" computing, but also co-located "fog" computing on larger platforms resulting in overall downlink of only the most critical information to the centralized ground station (cloud).  

## References