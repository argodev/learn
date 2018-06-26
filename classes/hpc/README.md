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




## Wednesday


## Thursday



