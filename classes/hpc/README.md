# Introduction to HPC
Building 5100, Room 128
June 26-28, 2018

## Tuesday

### Introduction to HPC
Ashley Barker & Tom Papatheodore

### Overview of High Performance Computing Resourses at the Oak Ridge Leadership Computing Facility
Tom Papatheodore

- Parnter with Liasons to help bridge the gap
- Competitive allocations (ALCC, etc.)
- Large allocations
- smallest allocation is 1 node (an AMD processor + a NVIDIA Kepler)
  - 16 core AMD Opteron with 32 GB RAM
  - Tesla K20X 6 GB RAM
  - connected via PCI 2.0 at 8 GB/s
- connected via Cray Gemini Interconnect... up to 6.4 GB/s
- Lustre File System (Atlas)
- 32 PB capacity
- 1 TB/s read/write (aggregate)
- HPSS (High Performance Storage System) (long term archival, disk/tape)
- Summit is coming in January 2019
  - IBM Power 9 processors
  - Each node has two Power9 procesors
  - Each node has 6 NVIDIA GV100 
  - 1600 GB SSD
  - 25 GB/s EDR IP (2 ports)
  - 512 GB DRAM
  - 96 GB HMB (cohereent shared memory
  - GPFS file system
  - 2.2 TB read/write
  - NVLink to CPU at up to 50 GB/s

Access
- Incite - approximately 50% of the allocation http://www.doeleadershipcomputing.org/
- ALCC - 20%
- ECP - 20%
- Director's Discretionary - 10% (1-5million core hours)

### Logging in to OLCF Machines

```bash
ssh csep03@home.ccs.ornl.gov

# or...
ssh csep03@titan.ccs.ornl.gov
```


### Intro to Unix
Bill Renaud

- Developed in 1969 by Ken Thompson and Dennis Ritchie
- Linux developed by Linux Torvalds in 1991
- GNU Project started by Richard Stallman in 1983 with the aim to provide a free, Unix-compatible OS
- Many of the world's most powerful computers use Linux + GNU software
- Kernel... the "main" OS program that's responsible for running the system. 
- Shell... A program that interfaces between the user and the kernel
- wildcards
  - * means match zero or more characters
  - ? matches 1 character


- ls helpers
  - -1 (one file per line)
  - -F (show file types)

- permissions
  - chmod ### file
  - user, group, other
  - read: 4
  - write: 2
  - execute: 1

### vim Text Editor

:set syntax=markdown

:set nu

### File Systems & Data Transfers
- Atlas `/luster/atlas`
- NFS `/ccs/home/csep03`
- HPSS

$MEMBERWORK
$PROJWORK
$WORLDWORK

data transfer nodes
dtn.ccs.ornl.gov


larger amounts of data
bbcp
fcp

scp, etc.

http://www.globus.org 


High Performance Storage System
Used via `hsi` and `htar`



### Programming Enviornment: modules, compilers, etc.

Programming Environment
- calable debuggers and analysis utilities
- Performance Math and Parallel libraries
- IO Service and Runtimes
- Compiler and programming model runtimes
- Compiler toolchains
- Userland Tools and utilities

The PE needs to be flexible and personalized
- this is *your* shell's build and runtime enviornment
- you load your own modules/settings when you are ready to
- /etc/profile (system defaults)
- user-specified defaults are in personal shell init scripts (in home dir)
- Using the enviornment module system (preferred)
- consistency and accuracy is important
- Titan uses TCL-based Enviornment Modules

```bash
. $MODULESHOME/init/$SHELL

module -d list
module avail
```

Cray programming enviornment
- PrgEnv-pgi, PrgEnv-gnu, PrgEnv-cray, PrgEnv-intel



### Batch Scheduler / Job Launcher

### Programming Basics: C, Fortran

### Hands-On (Unix, compile, launch)





## Wednesday


## Thursday



