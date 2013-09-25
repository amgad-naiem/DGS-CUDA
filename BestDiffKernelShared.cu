/*
 *  BestDiffKernel.cu
 *  heuristic CUDA
 *
 *  Created by Roberto Roverso on 25/08/09.
 *  Copyright 2009 Peerialism. All rights reserved.
 *
 */

#include <stdio.h>

//#define KERNEL_DEBUG

#define NEGINF -9999.0f
#define BLOCKSIZE 16
#define VS BLOCKSIZE*BLOCKSIZE

__device__ int findMaxLocal(float *resDiffs, int rowId, int num);

__device__ float mS(AijMatrix aijMatrix, int row, int col) {
	return aijMatrix.els[row * aijMatrix.width + col];
}

//__device__ inline void swap(int & a, int & b) {
//	// Alternative swap doesn't use a temporary register:
//	// a ^= b;
//	// b ^= a;
//	// a ^= b;
//
//	int tmp = a;
//	a = b;
//	b = tmp;
//}
//
//__device__ static void bitonicSort(int * values, int num) {
//	extern __shared__ int shared[];
//
//	const int tid = threadIdx.x;
//
//	// Copy input to shared mem.
//	shared[tid] = values[tid];
//
//	__syncthreads();
//
//	// Parallel bitonic sort.
//	for (int k = 2; k <= num; k *= 2)
//	{
//		// Bitonic merge:
//		for (int j = k / 2; j > 0; j /= 2)
//		{
//			int ixj = tid ^ j;
//
//			if (ixj > tid)
//			{
//				if ((tid & k) == 0)
//				{
//					if (shared[tid] > shared[ixj])
//					{
//						swap(shared[tid], shared[ixj]);
//					}
//				} else
//				{
//					if (shared[tid] < shared[ixj])
//					{
//						swap(shared[tid], shared[ixj]);
//					}
//				}
//			}
//
//			__syncthreads();
//		}
//	}
//
//	// Write result.
//	values[tid] = shared[tid];
//}

__device__ void bestDiffInternalRepeatShared(int rowId, float* diffs,
		int* bestChanges, AijMatrix A, int* persons, int* objects,
		bool* bannedSwitches, bool* clearedBannedSwitches, int* reset) {

	float maxDiff = 0.009f;

	if (persons[rowId] == -1 || reset[0] == 1)
	{
		int bestChangeCol = -1;
		int myCol = persons[rowId];
		if (myCol == -1)
			maxDiff = NEGINF;

		int myRow = rowId;
		int foundFreeObject = 0;
		int otherCol;

		int m1 = 0;
		int m2 = A.width;

		if (reset[0] == 0)
		{
			m1 = blockDim.x * blockIdx.x;
			m2 = blockDim.x * (blockIdx.x + 1);
		}

		for (otherCol = m1; otherCol < m2; otherCol++)
		{
			int otherRow = objects[otherCol];
			float difference = NEGINF;
			// Person is not assigned
			if (myCol == -1)
			{
				// Object considered not assigned
				if (otherRow == -1)
				{
					// happiness value for the per-obj association
					difference = mS(A, myRow, otherCol);
					if (foundFreeObject == 0)
					{
						maxDiff = difference;
						bestChangeCol = otherCol;
					}
					foundFreeObject = 1;
				} else if (foundFreeObject == 0 && !bannedSwitches[myRow
						* A.width + otherRow])
					// object is not free
					// Compare old case with new case
					// pos...better me
					// neg...better him
					difference = mS(A, myRow, otherCol) - mS(A, otherRow,
							otherCol);
			} else if (otherRow == -1)
				// Compare old case with new case
				difference = mS(A, myRow, otherCol) - mS(A, myRow, myCol);
			else if (mS(A, otherRow, myCol) != NEGINF)
			{
				// Both assigned
				// Switch improves overall happiness of the two assignments
				difference = mS(A, myRow, otherCol) + mS(A, otherRow, myCol)
						- (mS(A, myRow, myCol) + mS(A, otherRow, otherCol));
			}
			if (difference > maxDiff)
			{
				maxDiff = difference;
				bestChangeCol = otherCol;
			}
		}

#ifdef KERNEL_DEBUG
		printf("D%d -> %f\n", rowId, maxDiff);
#endif

		__syncthreads();

		if (maxDiff < 0)
			maxDiff = -maxDiff;
		if (maxDiff > 0.1 || myCol == -1)
		{
			if (bestChangeCol == -1)
			{
				if (clearedBannedSwitches[myRow])
				{
					persons[myRow] = -1;
					diffs[myRow] = NEGINF;
					bestChanges[myRow] = -1;
					return;
				}
				return;
			}
			if (myCol == -1)
				maxDiff = maxDiff * 1000;

			diffs[rowId] = maxDiff;
			bestChanges[rowId] = bestChangeCol;

			//			diffs[rowId].index = rowId;
			//			diffs[rowId].bestChange = bestChangeCol;
			//			diffs[rowId].type = 0;
			//			diffs[rowId].value = maxDiff;
			return;
		}
		diffs[rowId] = NEGINF;
		bestChanges[rowId] = -1;
		return;
	}

}

__global__ void bestDiffShared(float* diffs, int* bestChanges, AijMatrix A,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, int* reset, int blockSize, float* resDiffs) {

	int ty = threadIdx.y;
	int tx = threadIdx.x;

	int rowId = blockIdx.y * blockDim.y + ty;
	int colId = blockIdx.x * blockDim.x + tx;

	int rdI = rowId * A.width + colId;
	// General result for row (to be calculated later by gathering threads)
	//	float maxDiff = 0.009f;
	//	int bestChangeCol = -1;

	// Values needed for calculation, shared mem
	__shared__ int myCol[VS];
	__shared__ int myRow[VS];
	//int foundFreeObject = 0;
	__shared__ int otherCol[VS];
	__shared__ int otherRow[VS];
	__shared__ float mRmC[VS];
	__shared__ float mRoC[VS];
	__shared__ float oRmC[VS];
	__shared__ float oRoC[VS];
	__shared__ float difference[VS];
	__shared__ int banned;

	// INIT SHARED MEM
	int vI = ty * BLOCKSIZE + tx;
	myCol[vI] = persons[rowId];
	if (myCol[vI] == -1)
	{
		resDiffs[rdI] = NEGINF;
	}
	myRow[vI] = rowId;
	otherCol[vI] = colId;
	otherRow[vI] = objects[otherCol[vI]];
	myCol[vI] = persons[rowId];
	difference[vI] = NEGINF;
	mRmC[vI] = mS(A, myRow[vI], myCol[vI]);
	mRoC[vI] = mS(A, myRow[vI], otherCol[vI]);
	oRmC[vI] = mS(A, otherRow[vI], myCol[vI]);
	oRoC[vI] = mS(A, otherRow[vI], otherCol[vI]);

	if (bannedSwitches[myRow[vI] * A.width + otherRow[vI]])
	{
		banned = 1;
	} else
	{
		banned = 0;
	}

	__syncthreads();

	// WORKING ONLY ON SHARED MEM

	// Person is not assigned
	if (myCol[vI] == -1)
	{
		// Object considered not assigned
		if (otherRow[vI] == -1)
		{
			// happiness value for the per-obj association
			difference[vI] = mRoC[vI];
			// What happens if it finds multiple free objects
			//if (foundFreeObject == 0)
			//{
			//			maxDiff = difference;
			//			bestChangeCol = otherCol;
			//}
			//foundFreeObject = 1;
		} else if (/*foundFreeObject == 0 && */banned == 0)
			// object is not free
			// Compare old case with new case
			// pos...better me
			// neg...better him
			difference[vI] = mRoC[vI] - oRoC[vI];
	} else if (otherRow[vI] == -1)
	{
		// Compare old case with new case
		difference[vI] = mRoC[vI] - mRmC[vI];
	} else if (oRmC[vI] != NEGINF)
	{
		// Both assigned
		// Switch improves overall happiness of the two assignments
		difference[vI] = mRoC[vI] + oRmC[vI] - (mRmC[vI] + oRoC[vI]);
	}
	//if (difference > maxDiff)
	//{
	//	maxDiff = difference;
	//	bestChangeCol = otherCol;
	//}
	//}

//#ifdef KERNEL_DEBUG
//	printf("[%d][%d] I%d\n", rowId, colId,vI);
//	printf("myRow %d myCol %d ", myRow[vI], myCol[ty * BLOCKSIZE + tx]);
//	printf("otherRow %d otherCol %d ", otherRow[vI], otherCol[vI]);
//	printf("mRmC -> %f ", mRmC[vI]);
//	printf("mRoC -> %f ", mRoC[vI]);
//	printf("oRmC -> %f ", oRmC[vI]);
//	printf("oRoC -> %f \n", oRoC[vI]);
//	printf("D[%d,%d] -> %f\n", rowId, colId, resDiffs[rowId * A.width + colId]);
//#endif

	//int tId = threadIdx.x;

	resDiffs[rdI] = difference[vI];

	//	diffs[rowId] = maxDiff;

}

__global__ void findMaxShared(float* diffs, int* bestChanges, AijMatrix A,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, int* reset, int blockSize, float* resDiffs) {

	int tx = threadIdx.x;
	int rowId = blockIdx.x * blockDim.x + tx;
	// gathering by one thread per row
	//	if (colId == 0)
	//	{
	//		int maxIndex = 0;
	//		float p_max_value = resDiffs[rowId * A.width];
	//		for (int position = 1; position < A.width; ++position)
	//		{
	//			if (resDiffs[rowId * A.width + position] > p_max_value)
	//			{
	//				p_max_value = resDiffs[rowId * A.width + position];
	//				maxIndex = position;
	//			}
	//		}
	//		int bestChangeCol = 0;
	//		if (p_max_value > 0.009f)
	//		{
	//			bestChangeCol = maxIndex;
	//		} else
	//		{
	//			bestChangeCol = -1;
	//			p_max_value = 0.009f;
	//		}
	//		resDiffs[rowId * A.width] = p_max_value;
	//#ifdef KERNEL_DEBUG
	//	printf("curdiff[%d]=%f\n", rowId, resDiffs[rowId * A.width]);
	//#endif
	int bestChangeCol = findMaxLocal(resDiffs, rowId, A.width);

#ifdef KERNEL_DEBUG
	printf("Max for row %d=%f\n", rowId, resDiffs[rowId * A.width]);
#endif

	//		__syncthreads();
	//	}
	int myCol = persons[rowId];
	float curDiff = resDiffs[rowId * A.width];
#ifdef KERNEL_DEBUG
	printf("curdiff[%d]=%f\n", rowId, curDiff);
#endif
	if (curDiff < 0)
		curDiff = -curDiff;
	if (curDiff > 0.1 || myCol == -1)
	{
		if (bestChangeCol == -1)
		{
			if (clearedBannedSwitches[rowId])
			{
				// No suitable assignment due to banning
				persons[rowId] = -1;
				diffs[rowId] = NEGINF;
				bestChanges[rowId] = -1;
				return;
			}
			clearedBannedSwitches[rowId] = true;
			int x;
			for (x = 0; x < A.height; x++)
				bannedSwitches[rowId * A.width + x] = false;

			bestDiffInternalRepeatShared(rowId, diffs, bestChanges, A, persons,
					objects, bannedSwitches, clearedBannedSwitches, reset);
			return;
		}
		if (myCol == -1)
			curDiff = curDiff * 1000;

		diffs[rowId] = curDiff;
		bestChanges[rowId] = bestChangeCol;

#ifdef KERNEL_DEBUG
		printf("diff[%d]=%f bc[%d]=%d\n", rowId, curDiff, rowId, bestChangeCol);
#endif
		return;
	}
	// Difference not worth to consider
	diffs[rowId] = NEGINF;
	bestChanges[rowId] = -1;
	return;

}

__device__ int findMaxLocal(float *resDiffs, int rowId, int num) {
	int maxIndex = 0;
	float p_max_value = resDiffs[rowId * num];
#ifdef KERNEL_DEBUG
	printf("First v %f\n", p_max_value);
#endif
	for (int position = 1; position < num; ++position)
	{
		if (resDiffs[rowId * num + position] > p_max_value)
		{
			p_max_value = resDiffs[rowId * num + position];
			maxIndex = position;
		}
	}
	int bestChangeCol = 0;
	if (p_max_value > 0.009f)
	{
		bestChangeCol = maxIndex;
	} else
	{
		bestChangeCol = -1;
		p_max_value = 0.009f;
	}
	resDiffs[rowId * num] = p_max_value;
	return bestChangeCol;
}

