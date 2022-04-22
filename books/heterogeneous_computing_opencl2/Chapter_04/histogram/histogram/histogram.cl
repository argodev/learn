#define HIST_BINS 256

__kernel
void histogram(__global int *data, int numData, __global int *histogram) { 

	__local int localHistogram[HIST_BINS];
	int lid = get_local_id(0);
	int gid = get_global_id(0);

	/* initialize local histogram to zero */
	for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
		localHistogram[i] = 0;
	}

	/* wait until all work-items within the work-group  */
	/* have completed their stores */
	barrier(CLK_LOCAL_MEM_FENCE);

	/* compute local histogram */
	for (int i = gid; i < numData; i += get_global_size(0)) {
		atomic_add(&localHistogram[data[i]], 1);
	}

	/* wait until all work-items within the work-group have */
	/* completed their stores */
	barrier(CLK_LOCAL_MEM_FENCE);

	/* write the local histogram out to the */
	/* global histogram */
	for (int i = lid; i < HIST_BINS; i += get_local_size(0)) {
		atomic_add(&histogram[i], localHistogram[i]);
	}
}
