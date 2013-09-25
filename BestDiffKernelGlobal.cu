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

//__global__ void calculateHappiness(AijMatrix A, int* persons,
//		int numberOfPersons) {
//	int i;
//	float totalHappiness;
//	for (i = 0; i < numberOfPersons; i++)
//	{
//		if (persons[i] != -1)
//			totalHappiness += A.els[i + A.width + persons[i]];
//	}
//}

__device__ float m(AijMatrix aijMatrix, int row, int col) {
	return aijMatrix.els[row * aijMatrix.width + col];
}
__device__ void bestDiffInternalRepeat(int rowId, float* diffs,
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
					difference = m(A, myRow, otherCol);
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
					difference = m(A, myRow, otherCol) - m(A, otherRow,
							otherCol);
			} else if (otherRow == -1)
				// Compare old case with new case
				difference = m(A, myRow, otherCol) - m(A, myRow, myCol);
			else if (m(A, otherRow, myCol) != NEGINF)
			{
				// Both assigned
				// Switch improves overall happiness of the two assignments
				difference = m(A, myRow, otherCol) + m(A, otherRow, myCol)
						- (m(A, myRow, myCol) + m(A, otherRow, otherCol));
			}
			if (difference > maxDiff)
			{
				maxDiff = difference;
				bestChangeCol = otherCol;
			}
		}

#ifdef KERNEL_DEBUG
		printf("D%d -> %f\n",rowId,maxDiff);
#endif

//		__syncthreads();

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

__global__ void bestDiff(float* diffs, int* bestChanges, AijMatrix A,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, int* reset, int blockSize) {

	//__shared__ float diffs[BLOCKSIZE];

	float maxDiff = 0.009f;

	// Values needed for calculation

	int rowId = blockIdx.x * blockDim.x + threadIdx.x;

	if (persons[rowId] == -1 || reset[0] == 1)
	{
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
					difference = m(A, myRow, otherCol);
					if (foundFreeObject == 0)
					{
						maxDiff = difference;
						bestChanges[rowId] = otherCol;
					}
					foundFreeObject = 1;
				} else if (foundFreeObject == 0 && !bannedSwitches[myRow
						* A.width + otherRow])
					// object is not free
					// Compare old case with new case
					// pos...better me
					// neg...better him
					difference = m(A, myRow, otherCol) - m(A, otherRow,
							otherCol);
			} else if (otherRow == -1)
				// Compare old case with new case
				difference = m(A, myRow, otherCol) - m(A, myRow, myCol);
			else if (m(A, otherRow, myCol) != NEGINF)
			{
				// Both assigned
				// Switch improves overall happiness of the two assignments
				difference = m(A, myRow, otherCol) + m(A, otherRow, myCol)
						- (m(A, myRow, myCol) + m(A, otherRow, otherCol));
			}
			if (difference > maxDiff)
			{
				maxDiff = difference;
				bestChanges[rowId] = otherCol;
			}else{
				bestChanges[rowId]=-1;
			}

		}

		//int tId = threadIdx.x;

		diffs[rowId] = maxDiff;

#ifdef KERNEL_DEBUG
		printf("D%d -> %f\n",rowId,maxDiff);
#endif
	}
}

__global__ void findMax(float* diffs, int* bestChanges, AijMatrix A,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, int* reset, int blockSize) {

	int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	//if (rowId == 0 || (rowId % blockSize) == 0)
	//{
	//int blockStart = blockIdx.x * blockDim.x;
	//int blockEnd = blockIdx.x * blockDim.x + BLOCKSIZE;
	//int blockStart=0;
	//int blockEnd=A.height;

	//for (int var = blockStart; var < blockEnd; ++var)
	//{
	int myCol = persons[rowId];
	int bestChangeCol=bestChanges[rowId];
	float curDiff = diffs[rowId];
#ifdef KERNEL_DEBUG
	printf("curdiff[%d]=%f\n",rowId,curDiff);
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

			bestDiffInternalRepeat(rowId, diffs, bestChanges, A, persons,
					objects, bannedSwitches, clearedBannedSwitches, reset);
			return;
		}
		if (myCol == -1)
			curDiff = curDiff * 1000;

		diffs[rowId] = curDiff;
		bestChanges[rowId] = bestChangeCol;

#ifdef KERNEL_DEBUG
		printf("diff[%d]=%f\n",rowId,curDiff);
#endif
		return;
	}
	// Difference not worth to consider
	diffs[rowId] = NEGINF;
	bestChanges[rowId] = -1;
	return;

}

