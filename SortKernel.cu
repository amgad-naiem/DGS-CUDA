
#include <stdio.h>
//#include "Difference.c"
//#define KERNEL_DEBUG

#define NEGINF -9999.0f
#define BLOCKSIZE 16


__global__ void placeDifferencesByIndex(Difference* diffs, int* index,int max){

	int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	int idx=index[rowId];

	__shared__ Difference tempDiff[BLOCKSIZE];
	tempDiff[threadIdx.x]=diffs[idx];

#ifdef KERNEL_DEBUG
	printf("idxTm %d thread[%d]=%f\n",idx,rowId,tempDiff[threadIdx.x].value);
#endif

	__syncthreads();

	diffs[max-rowId-1]=tempDiff[threadIdx.x];

}
