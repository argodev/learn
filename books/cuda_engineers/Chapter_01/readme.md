# Chapter 1: First Steps

After a quick review of the introduction, I started reading this chapter. I immediately came to the first checkpoint: __Ensure you have a working CUDA development enviornment__ and therefore stopped and jumped to the appendicies to confirm my environment was ready. I quickly worked through [Appendix A: Hardware Setup](../Appendix_A/readme.md) followed by [Appendix B: Software Setup](../Appendix_B/readme.md). I then went on to perform the work in [Appendix C: Need-to-Know C Programming](../Appendix_C/readme.md) to further verify my system's setup. I wrapped up my walk through the appendicies with [Appendix D: CUDA Practicalities: Timing, Profiling, Error Handling, and Debugging](Appendix_D/readme.md). This last section walks through a number of common tasks one needs to know how to accomplish during CUDA development and I found this clear and helpful.


## Running some examples
Now that we confirmed that everything is working properly on our machine, and we've had a bit of an introduction to CUDA programming, we'll dive back into chapter 1 and follow along with some of the tests.

```bash
cd ~/cuda/NVIDIA_CUDA-9.0_Samples/bin/x86_64/linux/release
$ ./nbody

$ ./nbody -benchmark

GPU Device 0: "Quadro P3000" with compute capability 6.1

> Compute 6.1 CUDA device: [Quadro P3000]
10240 bodies, total time for 10 iterations: 11.587 ms
= 90.495 billion interactions per second
= 1809.902 single-precision GFLOP/s at 20 flops per interaction

$ ./nbody -benchmark -cpu

> Simulation with CPU
4096 bodies, total time for 10 iterations: 5077.967 ms
= 0.033 billion interactions per second
= 0.661 single-precision GFLOP/s at 20 flops per interaction

```

As recommended in the text, I also ran `./volumeRender` a bit and experimented with interacting with data in real time.

I went ahead and set up `dist_v1` and `dist_v2` as instructed, however, becuase I walked through Appendix C, these were already done and tested. I reconfirmed functionality and consistency between the two of them before wrapping up the first chapter.
