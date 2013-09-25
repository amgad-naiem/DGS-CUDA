#include <stdio.h>
#define NEGINF -9999.0f

__global__ void initialAssignment(float* diffs, int* bestChanges, AijMatrix A,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, int* reset, int* blockSize) {

	int row1;
	int col2;
	int curRow;

	int m1 = blockIdx.x * blockSize[0];

	int m2 = (blockIdx.x + 1) * blockSize[0];

	// Check of all have been assigned
#ifdef DEBUG
	printf("\n BS%d M1 %d, M2 %d\n", blockSize[0], m1, m2);
#endif
	for (int i = m1; i < m2; i++)
	{
		row1 = i;
		if (diffs[i] != NEGINF)
		{
#ifdef DEBUG
			printf("P%d != -1\n", i);
#endif
			col2 = bestChanges[row1]; // index of column of the best
			// cell in the row of difference
			if (objects[col2] == -1)
			{
				persons[row1] = col2;
				objects[col2] = row1;
			} else
			{

				curRow = objects[col2];
				if (diffs[curRow] < diffs[row1])
				{
					//Swap
					if (curRow != -1)
					{
#ifdef DEBUG
						printf("P%d == -1\n", curRow);
#endif
						persons[curRow] = -1;
						bannedSwitches[row1 * A.width + curRow] = true;
						reset[0] = 1;
					}
					persons[row1] = col2;
					objects[col2] = row1;
				}
			}
		}
#ifdef DEBUG
		printf("P%d->%d\n", i, persons[i]);
#endif
	}
}

