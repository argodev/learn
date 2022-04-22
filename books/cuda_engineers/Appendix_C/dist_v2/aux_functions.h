#ifndef AUX_FUNCTIONS_H
#define AUX_FUNCTIONS_H

// function to scale input on interval [0,1]
float scale(int i, int n);

// compute the distance between two points on a line
float distance(float x1, float x2);

// compute scale distance for an array of input values
void distanceArray(float *out, float *in, float ref, int n);

#endif
