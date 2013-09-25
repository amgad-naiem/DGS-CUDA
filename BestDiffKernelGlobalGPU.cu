/*
 *  BestDiffKernel.cu
 *  heuristic CUDA
 *
 *  Created by Roberto Roverso on 25/08/09.
 *  Copyright 2009 Peerialism. All rights reserved.
 *
 */

#include <stdio.h>
//#include "Difference.c"
//#define KERNEL_DEBUG

#define NEGINF -9999.0f
#define BLOCKSIZE 16

__device__ float m(AijMatrix aijMatrix, int row, int col) {
	return aijMatrix.els[row * aijMatrix.width + col];
}

__device__ void setEmptyDiff(Difference* diff) {
	diff->index = -1;
	diff->myAssigned = -1;
	diff->bestChangeAssigned = -1;
}

__device__ void setDifferenceProps(Difference* diff, int idx, int bestChange,
		int myAssigned, int bestChangeAssigned, float value) {
	diff->index = idx;
	diff->bestChange = bestChange;
	//	diff->type = type;
	diff->myAssigned = myAssigned;
	diff->bestChangeAssigned = bestChangeAssigned;
	diff->value = value;
}

__device__ void bestDiffInternalRepeat(int rowId, Difference* diffs,float* srtDiffs,
		AijMatrix A, int* persons, int* objects, char* bannedSwitches,
		char* clearedBannedSwitches) {

	float maxDiff = 0.009f;

	//	if (persons[rowId] == -1 || reset[0] == 1)
	//	{
	int bestChangeCol = -1;
	int myCol = persons[rowId];
	if (myCol == -1)
		maxDiff = NEGINF;

	int myRow = rowId;
	int foundFreeObject = 0;
	int otherCol;

	int m1 = 0;
	int m2 = A.width;

	//		if (reset[0] == 0)
	//		{
	//
	//			m1 = blockDim.x * blockIdx.x;
	//			m2 = blockDim.x * (blockIdx.x + 1);
	//		}

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
			} else if (foundFreeObject == 0 && !bannedSwitches[myRow * A.width
					+ otherRow])
				// object is not free
				// Compare old case with new case
				// pos...better me
				// neg...better him
				difference = m(A, myRow, otherCol) - m(A, otherRow, otherCol);
		} else if (otherRow == -1)
			// Compare old case with new case
			difference = m(A, myRow, otherCol) - m(A, myRow, myCol);
		else if (m(A, otherRow, myCol) != NEGINF)
		{
			// Both assigned
			// Switch improves overall happiness of the two assignments
			difference = m(A, myRow, otherCol) + m(A, otherRow, myCol) - (m(A,
					myRow, myCol) + m(A, otherRow, otherCol));
		}
		if (difference > maxDiff)
		{
			maxDiff = difference;
			bestChangeCol = otherCol;
		}
	}

//#ifdef KERNEL_DEBUG
//	printf("D%d -> %f\n",rowId,maxDiff);
//#endif

	if (maxDiff < 0)
	{
		maxDiff = -maxDiff;
	}
	if (maxDiff > 0.1 || myCol == -1)
	{
		if (bestChangeCol == -1)
		{
			if (clearedBannedSwitches[myRow])
			{
				persons[myRow] = -1;
				setEmptyDiff(&diffs[myRow]);
				srtDiffs[rowId]=0.0f;
				return;
			}
			return;
		}
		if (myCol == -1)
			maxDiff = maxDiff * 1000;

		setDifferenceProps(&diffs[rowId], rowId, bestChangeCol, persons[rowId],
				objects[bestChangeCol], maxDiff);
		srtDiffs[rowId]=maxDiff;
		return;
	}
	setEmptyDiff(&diffs[myRow]);
	srtDiffs[rowId]=0.0f;
	return;
	//	}

}

__global__ void evaluateDiff(AijMatrix A, Difference* diffs,float* srtDiffs, int* persons,
		int* objects, char* bannedSwitches, char* clearedBannedSwitches) {

	//__shared__ float diffs[BLOCKSIZE];

	float maxDiff = 0.009f;

	// Values needed for calculation

	int rowId = blockIdx.x * blockDim.x + threadIdx.x;

	//	if (persons[rowId] == -1 || reset[0] == 1)
	//	{
	int myCol = persons[rowId];
	if (myCol == -1)
		maxDiff = NEGINF;

	int myRow = rowId;
	int foundFreeObject = 0;
	int otherCol;
	int bestChangeCol=-1;

	int m1 = 0;
	int m2 = A.width;

	//		if (reset[0] == 0)
	//		{
	//
	//			m1 = blockDim.x * blockIdx.x;
	//			m2 = blockDim.x * (blockIdx.x + 1);
	//		}

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
			} else if (foundFreeObject == 0 && !bannedSwitches[myRow * A.width
					+ otherRow])
				// object is not free
				// Compare old case with new case
				// pos...better me
				// neg...better him
				difference = m(A, myRow, otherCol) - m(A, otherRow, otherCol);
		} else if (otherRow == -1)
			// Compare old case with new case
			difference = m(A, myRow, otherCol) - m(A, myRow, myCol);
		else if (m(A, otherRow, myCol) != NEGINF)
		{
			// Both assigned
			// Switch improves overall happiness of the two assignments
			difference = m(A, myRow, otherCol) + m(A, otherRow, myCol) - (m(A,
					myRow, myCol) + m(A, otherRow, otherCol));
		}
		if (difference > maxDiff)
		{
			maxDiff = difference;
			bestChangeCol = otherCol;
		}

	}

	//int tId = threadIdx.x;

//#ifdef KERNEL_DEBUG
//	printf("D%d -> %f\n",rowId,maxDiff);
//#endif
	//	}

	//	int rowId = blockIdx.x * blockDim.x + threadIdx.x;
	//if (rowId == 0 || (rowId % blockSize) == 0)
	//{
	//int blockStart = blockIdx.x * blockDim.x;
	//int blockEnd = blockIdx.x * blockDim.x + BLOCKSIZE;
	//int blockStart=0;
	//int blockEnd=A.height;

	//for (int var = blockStart; var < blockEnd; ++var)
	//{
	if (maxDiff < 0)
		maxDiff = -maxDiff;
	if (maxDiff > 0.1 || myCol == -1)
	{
		if (bestChangeCol == -1)
		{
			if (clearedBannedSwitches[rowId])
			{
				// No suitable assignment due to banning
				persons[rowId] = -1;
				setEmptyDiff(&diffs[myRow]);
				srtDiffs[rowId]=0.0f;
				return;
			}
			clearedBannedSwitches[rowId] = 1;
			int x;
			for (x = 0; x < A.height; x++)
				bannedSwitches[rowId * A.width + x] = 0;

			bestDiffInternalRepeat(rowId, diffs,srtDiffs, A, persons,
					objects, bannedSwitches, clearedBannedSwitches);
			return;
		}
		if (myCol == -1)
			maxDiff = maxDiff * 1000;

		setDifferenceProps(&diffs[rowId], rowId, bestChangeCol, persons[rowId],
				objects[bestChangeCol], maxDiff);
		srtDiffs[rowId]=maxDiff;
		//		curDiff.index = i;
		//		curDiff.bestChange = bestChangeCol;
		//		curDiff.type = 0;
		//		curDiff.myAssigned = persons[i];
		//		curDiff.bestChangeAssigned = objects[bestChangeCol];
		//		curDiff.value = maxDiff;

//#ifdef KERNEL_DEBUG
//		printf("srtDiff[%d]=%f\n",rowId,srtDiffs[rowId]);
//#endif
		return;
	}
	// Difference not worth to consider
	setEmptyDiff(&diffs[myRow]);
	srtDiffs[rowId]=0.0f;
	return;

}

