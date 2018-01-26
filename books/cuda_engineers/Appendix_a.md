# Appendix A: Hardware Setup

## Checking for an NVIDIA GPU: Linux

```bash
$ lspci | grep -i nvidia
01:00.0 VGA compatible controller: NVIDIA Corporation GP104GLM [Quadro P3000 Mobile] (rev a1)
```


## Determining Compute Capability
Went to [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) to check on the status of my GPU.
`CUDA-Enabled Quadro Products` --> P3000 --> 6.1


## What is included
I poked around on the nvidia website for information about my GPU. I found the [Quadro Product Comparison](https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/documents/quadro-mobile-pro-graphics-line-card-us-r1-hr.pdf) pdf document and learned the following about the P3000:

| Metric | Value |
|:-------|------:|
|CUDA Processing Cores | 1,280 |
| GPU Memory | 6 GB |
| Memory Bandwidth | 168 GBps |
| Memory Type | GDDR5 |
| Memory Interface | 192-bit |
| TGP Max Power Consumption | 75W |
| OpenGL | 4.5 |
| Shader Model | 5.1 |
| DirectX | 12 |
| PCIe | 3 |
| Floating Point Performance | 3.1 |



test
