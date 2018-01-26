#include "aux_functions.cpp"
#define N 64

int main() {
    // create two arrays of N floats (initialized to 0.0)
    // we will overwrite these values to store inputs and outputs
    float in[N] = {0.0f};
    float out[N] = {0.0f};

    // chose a reference value from which distances are measured
    const float ref = 0.5f;

    // iteration loop computes array of scaled input vlues.
    for (int i = 0; i < N; ++i) {
        in[i] = scale(i, N);
    }

    // single function call to compute entire distance array
    distanceArray(out, in, ref, N);

    return 0;
}
