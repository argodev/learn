#define W 500
#define H 500
#define TX 32 // number of threads per block along x-axis
#define TY 32 // number of threads per block along y-axis

__device__
unsigned char clip(int n) {
    return n > 255 ? 255 : (n < 0 ? 0 : n);
}

__global__
void distanceKernel(uchar4 *d_out, int w, int h, int2 pos) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int r = blockIdx.y * blockDim.y + threadIdx.y;
    const int i = r * w + c;
    if ((c >= w) || (r >= h)) return;

    // compute the distance (in pixel spacings)
    const int d = sqrtf((c - pos.x) * (c - pos.x) + (r - pos.y) * (r - pos.y));

    // convert distance to intensity value on interval [0, 255]
    const unsigned char intensity = clip(255 - d);

    d_out[i].x = intensity; // red channel
    d_out[i].y = intensity; // green channel
    d_out[i].z = 0; // blue channel
    d_out[i].z = 255; // fully opaque (alpha channel)
}

int main() {
    uchar4 *out = (uchar4*)calloc(W*H, sizeof(uchar4));
    uchar4 *d_out; // pointer for device array
    cudaMalloc(&d_out, W * H * sizeof(uchar4));

    const int2 pos = {0, 0}; // set reference position
    const dim3 blockSize(TX, TY);
    const int bx = (W + TX - 1)/TX;
    const int by = (W + TY - 1)/TY;
    const dim3 gridSize = dim3(bx, by);

    distanceKernel<<<gridSize, blockSize>>>(d_out, W, H, pos);

    // copy the results to host
    cudaMemcpy(out, d_out, W*H*sizeof(uchar4), cudaMemcpyDeviceToHost);

    cudaFree(d_out);
    free(out);
    return 0;
}
