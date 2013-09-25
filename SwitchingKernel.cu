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


__device__ void removeDifferenceG(AijMatrix A, Difference* differences, int id) {
	setEmptyDiff(&differences[id]);
	int i;
	for (i = id + 1; i < A.height + A.width; i++)
	{
		if (differences[i].value != -1)
		{
			differences[i - 1] = differences[i];
		} else
		{
			break;
		}
	}
}

__device__ void bestDiffInternalRepeatOne(int rowId, Difference* diffs,
		float* srtDiffs, AijMatrix A, int* persons, int* objects,
		bool* bannedSwitches, bool* clearedBannedSwitches, Difference ret) {

	float maxDiff = 0.009f;

	int bestChangeCol = -1;
	int myCol = persons[rowId];
	if (myCol == -1)
		maxDiff = NEGINF;

	int myRow = rowId;
	int foundFreeObject = 0;
	int otherCol;

	int m1 = 0;
	int m2 = A.width;

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
				ret.index = -1;
				ret.myAssigned = -1;
				ret.bestChangeAssigned = -1;
				return;
			}
		}
		if (myCol == -1)
			maxDiff = maxDiff * 1000;

		ret.index = rowId;
		ret.bestChange = bestChangeCol;
		ret.type = 0;
		ret.value = maxDiff;
		return;
	}

	ret.index = -1;
	ret.myAssigned = -1;
	ret.bestChangeAssigned = -1;
	return;
}

__device__ void bestDiffOne(AijMatrix A, Difference* diffs, float* srtDiffs,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, int rowId, Difference ret) {

	float maxDiff = 0.009f;

	int myCol = persons[rowId];
	if (myCol == -1)
		maxDiff = NEGINF;

	int myRow = rowId;
	int foundFreeObject = 0;
	int otherCol;
	int bestChangeCol = -1;

	int m1 = 0;
	int m2 = A.width;

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
				ret.index = -1;
				ret.myAssigned = -1;
				ret.bestChangeAssigned = -1;
				return;
				//return myDifference;
			}
			clearedBannedSwitches[rowId] = true;
			int x;
			for (x = 0; x < A.height; x++)
				bannedSwitches[rowId * A.width + x] = false;

			bestDiffInternalRepeatOne(rowId, diffs, srtDiffs, A, persons,
					objects, bannedSwitches, clearedBannedSwitches, ret);
			return;
		}
		if (myCol == -1)
			maxDiff = maxDiff * 1000;

		ret.index = rowId;
		ret.bestChange = bestChangeCol;
		ret.type = 0;
		ret.value = maxDiff;
		return;
	}
	// Difference not worth to consider
	ret.index = -1;
	ret.myAssigned = -1;
	ret.bestChangeAssigned = -1;
	return;
}

__device__ void addRowBestDifference(AijMatrix A, Difference* differences,
		Difference* rowDifferences, float* srtDiffs, int* persons,
		int* objects, bool* bannedSwitches, bool* clearedBannedSwitches,
		int rowId, int sort, Difference ret) {
	bestDiffOne(A, differences, srtDiffs, persons, objects, bannedSwitches,
			clearedBannedSwitches, rowId, ret);
	Difference myDifference = ret;
	int numberOfPersons = A.height;
	int numberOfObjects = A.width;
	if (myDifference.index != -1)
	{
		myDifference.myAssigned = persons[rowId];
		myDifference.bestChangeAssigned = objects[myDifference.bestChange];
		if (sort == 0)
		{
			differences[rowId] = myDifference;
		} else
		{
			int i;
			float maxDiff = myDifference.value;
			for (i = 0; i < numberOfPersons + numberOfObjects; i++)
			{
				if (maxDiff > differences[i].value)
				{
					Difference currentDiff = myDifference;
					int j;
					for (j = i; j < numberOfObjects + numberOfPersons; j++)
					{
						if (differences[j].value != -1)
						{
							Difference temp = differences[j];
							differences[j] = currentDiff;
							currentDiff = temp;
						} else
						{
							differences[j] = currentDiff;
							break;
						}
					}
					break;
				}
			}

			//addSortedDifference(myDifference);
		}
		rowDifferences[rowId] = myDifference;
	}
}

__device__ void addColBestDifferenceG(AijMatrix A, Difference* differences,
		Difference* columnDifferences, int* persons, int* objects, int colId,
		int sort) {

	if (colId == -1 || objects[colId] == -1)
		return;
	float maxDiff = 0.009;
	int bestChangeRow = -1;
	int myRow = objects[colId];
	int myCol = colId;
	int otherRow;
	int numberOfPersons = A.height;
	int numberOfObjects = A.width;
	for (otherRow = 0; otherRow < numberOfPersons; otherRow++)
	{
		int otherCol = persons[otherRow];
		if (otherCol == -1)
			continue;
		if (m(A, otherRow, myCol) != NEGINF && m(A, myRow, otherCol) != NEGINF)
		{
			float difference = m(A, myRow, otherCol) + m(A, otherRow, myCol)
					- (m(A, myRow, myCol) + m(A, otherRow, otherCol));
			if (difference > maxDiff)
			{
				maxDiff = difference;
				bestChangeRow = otherRow;
			}
		}
	}
	if (maxDiff > 0.1)
	{
		Difference myDifference;
		myDifference.index = colId;
		myDifference.bestChange = bestChangeRow;
		myDifference.type = 1;
		myDifference.value = maxDiff;

		myDifference.myAssigned = objects[colId];
		myDifference.bestChangeAssigned = persons[bestChangeRow];
		if (sort == 0)
		{
			differences[numberOfPersons + colId] = myDifference;
		} else
		{
			int i;
			float maxDiff = myDifference.value;
			for (i = 0; i < numberOfPersons + numberOfObjects; i++)
			{
				if (maxDiff > differences[i].value)
				{
					Difference currentDiff = myDifference;
					int j;
					for (j = i; j < numberOfObjects + numberOfPersons; j++)
					{
						if (differences[j].value != -1)
						{
							Difference temp = differences[j];
							differences[j] = currentDiff;
							currentDiff = temp;
						} else
						{
							differences[j] = currentDiff;
							break;
						}
					}
					break;
				}
			}
			//addSortedDifference(myDifference);
		}
		columnDifferences[colId] = myDifference;
	}
}

__global__ void first(AijMatrix A, Difference* differences,
		Difference* rowDifferences, Difference* columnDifferences,
		float* srtDiffs, int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, Difference ret) {

	int rowId = blockIdx.x * blockDim.x + threadIdx.x;

	if (rowId == 0)
	{
		int numberOfPersons = A.height;
		int numberOfObjects = A.width;
		int switchedRows[2];
		int switchedColumns[2];
		while (differences[1].index > 1)
		{

			Difference myDiff = differences[0];

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

				if (col1 != myDiff.myAssigned || row2
						!= myDiff.bestChangeAssigned)
				{
					diffCheck = -1.0;
				} else if (row2 == -1)
				{
					diffCheck = m(A, row1, col2) - m(A, row1, col1);
				} else
				{
					diffCheck = m(A, row1, col2) + m(A, row2, col1) - (m(A,
							row1, col1) + m(A, row2, col2));
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
				if (row1 != myDiff.myAssigned || col2
						!= myDiff.bestChangeAssigned)
					diffCheck = -1.0f;
				else
					diffCheck = m(A, row1, col2) + m(A, row2, col1) - (m(A,
							row1, col1) + m(A, row2, col2));
			}
			//printf("DiffCheck: \n");
			// We need to check that our previous calculation still holds
			// It may not due to second order effects
			if (diffCheck <= 0)
			{
				if (myDiff.type == 0)
					setEmptyDiff(&rowDifferences[myDiff.index]);
				else
					setEmptyDiff(&columnDifferences[myDiff.index]);
				removeDifferenceG(A, differences, 0);
				continue;
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
			switchedRows[0] = row1;
			switchedRows[1] = row2;
			switchedColumns[0] = col1;
			switchedColumns[1] = col2;
			int i;

			for (i = 0; i < 2; i++)
			{
				if (columnDifferences[switchedColumns[i]].index != -1)
				{
					Difference toRemove = columnDifferences[switchedColumns[i]];
					int z;
					for (z = 1; z < numberOfObjects + numberOfPersons; z++)
					{
						Difference toCheck = differences[z];
						if (toCheck.index == -1)
							break;
						if (toCheck.index == toRemove.index && toCheck.type
								== toRemove.type)
						{
							removeDifferenceG(A, differences, z);
							break;
						}
					}
					setEmptyDiff(&columnDifferences[switchedColumns[i]]);
				}
				addColBestDifferenceG(A, differences, columnDifferences,
						persons, objects, switchedColumns[i], 1);
			}
			for (i = 0; i < 2; i++)
			{
				if (rowDifferences[switchedRows[i]].index != -1)
				{
					Difference toRemove = rowDifferences[switchedRows[i]];
					int z;
					for (z = 1; z < numberOfObjects + numberOfPersons; z++)
					{
						Difference toCheck = differences[z];
						if (toCheck.index == -1)
							break;
						if (toCheck.index == toRemove.index && toCheck.type
								== toRemove.type)
						{
							removeDifferenceG(A, differences, z);
							break;
						}
					}
					setEmptyDiff(&rowDifferences[switchedRows[i]]);
				}
				addRowBestDifference(A, differences, rowDifferences, srtDiffs,
						persons, objects, bannedSwitches,
						clearedBannedSwitches, rowId, 1, ret);

			}
		}
		//__syncthreads();
	}

}

