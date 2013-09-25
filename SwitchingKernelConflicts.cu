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

__device__ float mA(AijMatrix aijMatrix, int row, int col) {
	return aijMatrix.els[row * aijMatrix.width + col];
}

__global__ void switching(AijMatrix A, Difference* differences, bool* conflicts,
		int* persons, int* objects, bool* changedRows, bool* changedCols,
		float* srtDiffs, int m) {

	int rowId = blockIdx.x * blockDim.x + threadIdx.x;

#ifdef KERNEL_DEBUG
	printf("row id -> %d\n",rowId);
#endif

	if (conflicts[rowId])
	{
		//		if (rowId <= m)
		//		{
		if (differences[rowId].type == 0)
			changedRows[rowId] = true;
		else
			changedCols[rowId] = true;
		//		}
	} else
	{

		//		if (rowId <= m)
		//		{
#ifdef KERNEL_DEBUG
		printf("row id in -> %d\n",rowId);
#endif
		//			Difference myDiff = differences[i];
		//			if (conflicts[i])
		//			{
		//				if (myDiff.type == 0)
		//					changedRows[i] = true;
		//				else
		//					changedCols[i] = true;
		//			} else
		//			{
		//				m = m + 1;
		//			}

		Difference myDiff = differences[rowId];

		int row1, row2, col1, col2;
		// Here I need to retrieve the 2 columns and 2 rows that will be
		// altered due to switch ...
		// to not reUpdate all the differences in the Tree
		float diffCheck;
		if (myDiff.type == 0)
		{ // Found in row..i.e. switching happens
			//printf("Switching on row: \n");
			// along columns
			row1 = myDiff.index; // index of row of the difference
			//printf("index: \n");
			col1 = persons[row1]; // index of column of the chosen
			// cell in the row of difference
			col2 = myDiff.bestChange; // index of column of the best
			//printf("bc: \n");
			// cell in the row of difference
			row2 = objects[col2]; // index of row of the chosen in the
			// column of the best cell in the
			// difference row
			//printf("ma %d, bca %d",myDiff.myAssigned,myDiff.bestChangeAssigned);

			if (col1 != myDiff.myAssigned || row2 != myDiff.bestChangeAssigned)
			{
				diffCheck = -1.0;
			} else if (row2 == -1)
			{
				diffCheck = mA(A, row1, col2) - mA(A, row1, col1);
			} else
			{
				diffCheck = mA(A, row1, col2) + mA(A, row2, col1) - (mA(A,
						row1, col1) + mA(A, row2, col2));
			}
		} else
		{

			//printf("Switching on column: \n");
			col1 = myDiff.index; // index of column of the difference
			row1 = objects[col1]; // index of row of the chosen cell
			// in the column of difference
			row2 = myDiff.bestChange; // index of row of the best cell
			// in the column of difference
			col2 = persons[row2]; // index of column of the chosen in
			// the row of the best cell in the
			// difference column
			if (row1 != myDiff.myAssigned || col2 != myDiff.bestChangeAssigned)
				diffCheck = -1.0f;
			else
				diffCheck = mA(A, row1, col2) + mA(A, row2, col1) - (mA(A,
						row1, col1) + mA(A, row2, col2));
		}
		//printf("DiffCheck: \n");
		// We need to check that our previous calculation still holds
		// It may not due to second order effects
		if (diffCheck <= 0)
		{
			if (rowId == 0)
			{
				for (int i = 0; i < A.height + A.width; i++)
				{
					srtDiffs[i] = 0.0f;
				}
			}
			return;
		}
#ifdef KERNEL_DEBUG
		printf("DiffCheck -> %f\n",diffCheck);
#endif
		persons[row1] = col2;
		if (row2 != -1)
		{
			// if (col1 == -1)
			// bannedSwitches[row1].add(row2);
			persons[row2] = col1;
		}
		// if (col1 != -1)
		objects[col1] = row2;
		objects[col2] = row1;
		// if (col1 == -1 && row2 == -1)
		// return;

		// System.out.println("Happiness after switch:
		// "+calculateTotalHappiness());

		// Now we update the modified rows and columns
		changedRows[row1] = true;
		changedRows[row2] = true;
		changedCols[col1] = true;
		changedCols[col2] = true;

		//		}
	}
	if (rowId == 0)
	{
		for (int i = 0; i < A.height + A.width; i++)
		{
			srtDiffs[i] = 0.0f;
		}
	}

	__syncthreads();
}

