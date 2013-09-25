/*
 * AssignmentEngine.c
 *
 *  Created on: Jul 15, 2009
 *      Author: amgadnaiem
 */

//#include <jni.h>
#include "Global.h"
#ifdef MAC
#include "sys/malloc.h" // mac os x
#else
#include "malloc.h" // linux, windows
#endif
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

int count = 0;

float calculateTotalHappiness();
float calculateIterationHappiness(int* personsLocal);
Difference getRowBestDifference(int);
void addColBestDifference(int, int);
void addRowBestDifference(int, int);
void addRowBestDifferenceMod(int, int, Difference*);
//void evaluateDifferences();
//void fastInitialAssignment();
//void fastInitialAssignmentWithInitial(int initialAssignmet[]);
//void smartInitialAssignment();
//void smartInitialAssignmentWithInitial(int initialAssignmet[]);
//void enhanceBySwitching();
void removeDifference(int);
void addSortedDifference(Difference);
void addSortedDifferenceMod(Difference *myDifference);
int compare_differences(const void *, const void *);
int compare_differences_increasing(const void *a1, const void *b1);
static void sort(char *array, size_t size, int(*cmp)(void*, void*), int begin,
		int end);

int compare_differences(const void *a1, const void *b1) {
	Difference a = *(Difference*) a1;
	Difference b = *(Difference*) b1;
	if (a.index == -1)
		return 1;
	if (b.index == -1)
		return -1;
	float diff = a.value - b.value;
	if (diff > 0)
		return -1;
	else if (diff < 0)
		return 1;
	else {
		int idx = a.index - b.index;
		if (idx > 0)
			return -1;
		else if (idx < 0)
			return 1;
		return 0;
	}
}

//void fastInitialAssignment() {
//	int i;
//	for (i = 0; i < numberOfPersons; i++)
//		if (persons[i] == -1) {
//			int curRow = i;
//			Difference myDiff = getRowBestDifference(curRow);
//			int col2 = myDiff.bestChange;
//			if (objects[col2] == -1) {
//				persons[curRow] = col2;
//				objects[col2] = curRow;
//			}
//		}
//}
//
//void fastInitialAssignmentWithInitial(int initialAssignmet[]) {
//	int i;
//	for (i = 0; i < numberOfPersons; i++)
//		if (persons[i] == -1) {
//			if (initialAssignmet[i] != -1 && objects[initialAssignmet[i]] == -1
//				&& aijMatrix[i][initialAssignmet[i]] != negInf) {
//				persons[i] = initialAssignmet[i];
//				objects[initialAssignmet[i]] = i;
//			} else {
//				int curRow = i;
//				Difference myDiff = getRowBestDifference(curRow);
//				int col2 = myDiff.bestChange;
//				if (objects[col2] == -1) {
//					persons[curRow] = col2;
//					objects[col2] = curRow;
//				}
//			}
//		}
//}

//void smartInitialAssignment() {
//	int row1, curRow, col2, i;
//	//  time_t start = time(NULL);
//
//	for (i = 0; i < numberOfPersons; i++) {
//
//		//Kernel here
//
//		curRow = i;
//		while (curRow != -1) {
//			Difference myDiff = getRowBestDifference(curRow);
//			if (myDiff.index != -1) {
//				row1 = myDiff.index; // index of row of the difference
//				col2 = myDiff.bestChange; // index of column of the best
//				// cell in the row of difference
//				curRow = objects[col2]; // index of row of the chosen in the
//				// column of the best cell in the
//				// difference row
//				persons[row1] = col2;
//				objects[col2] = row1;
//				if (curRow != -1) {
//					persons[curRow] = -1;
//					bannedSwitches[row1][curRow] = 1;
//				}
//			}
//		}
//		//		if (time(NULL) - start > initializationTimeLim) {
//		//			fastInitialAssignment();
//		//			break;
//		//		}
//	}
//	//for (i = 0 ; i < numberOfPersons; i++)
//	//	printf("person %d : %d ,", (i+1), persons[i] );
//}
//
//
//void smartInitialAssignmentWithInitial(int initialAssignmet[]) {
//	int row1, curRow, col2, i;
//	//  time_t start = time(NULL);
//	for (i = 0; i < numberOfPersons; i++) {
//		if (initialAssignmet[i] != -1 && objects[initialAssignmet[i]] == -1 &&
//			aijMatrix[i][initialAssignmet[i]] != negInf) {
//			persons[i] = initialAssignmet[i];
//			objects[initialAssignmet[i]] = i;
//		} else {
//			curRow = i;
//			while (curRow != -1) {
//				Difference myDiff = getRowBestDifference(curRow);
//				if (myDiff.index != -1) {
//					row1 = myDiff.index; // index of row of the
//					// difference
//					col2 = myDiff.bestChange; // index of column of the
//					// best cell in the row of
//					// difference
//					curRow = objects[col2]; // index of row of the chosen in
//					// the column of the best cell
//					// in the difference row
//					persons[row1] = col2;
//					objects[col2] = row1;
//					if (curRow != -1) {
//						persons[curRow] = -1;
//						bannedSwitches[row1][curRow] = 1;
//					}
//				}
//			}
//			//		if (time(NULL) - start > initializationTimeLim) {
//			//			fastInitialAssignment();
//			//			break;
//			//		}
//		}
//	}
//}

float calculateTotalHappiness() {
	float totalHappiness = 0;
	int i;
	for (i = 0; i < numberOfPersons; i++) {
		if (persons[i] != -1)
			totalHappiness += aijMatrix[i][persons[i]];
	}
	return totalHappiness;
}

float calculateIterationHappiness(int* personsLocal) {
	float totalHappiness = 0;
	int i;
	for (i = 0; i < numberOfPersons; i++) {
		if (personsLocal[i] != -1)
			totalHappiness += aijMatrix[i][personsLocal[i]];
	}
	return totalHappiness;
}

//void enhanceBySwitching() {
//	float newTotalHappiness, oldTotalHappiness;
//	// int counter = 1;
//	//  time_t start = time(NULL);
//	while (1) {
//		// System.out.println(counter++);
//		oldTotalHappiness = calculateTotalHappiness();
//		evaluateDifferences();
//		int switchedRows[2];
//		int switchedColumns[2];
//		qsort (differences, numberOfObjects + numberOfPersons, sizeof(Difference), compare_differences);
//		//int l;
//		//for (l = 0; l < numberOfObjects + numberOfPersons; l++)
//		//	printf("%f ,", differences[l].value);
//		while (differences[0].index > 0) {
//			Difference myDiff = differences[0];
//
//			int row1, row2, col1, col2;
//			// Here I need to retrieve the 2 columns and 2 rows that will be
//			// altered due to switch ...
//			// to not reUpdate all the differences in the Tree
//			float diffCheck;
//			if (myDiff.type == 0) { // Found in row..i.e. switching happens
//				// along columns
//				row1 = myDiff.index; // index of row of the difference
//				col1 = persons[row1]; // index of column of the chosen
//				// cell in the row of difference
//				col2 = myDiff.bestChange; // index of column of the best
//				// cell in the row of difference
//				row2 = objects[col2]; // index of row of the chosen in the
//				// column of the best cell in the
//				// difference row
//				if (col1 != myDiff.myAssigned || row2 != myDiff.bestChangeAssigned) {
//					diffCheck = -1.0;
//				} else if (row2 == -1) {
//					diffCheck = aijMatrix[row1][col2] - aijMatrix[row1][col1];
//				} else {
//					diffCheck = aijMatrix[row1][col2] + aijMatrix[row2][col1] - (aijMatrix[row1][col1] + aijMatrix[row2][col2]);
//				}
//			} else {
//				col1 = myDiff.index; // index of column of the difference
//				row1 = objects[col1]; // index of row of the chosen cell
//				// in the column of difference
//				row2 = myDiff.bestChange; // index of row of the best cell
//				// in the column of difference
//				col2 = persons[row2]; // index of column of the chosen in
//				// the row of the best cell in the
//				// difference column
//				if (row1 != myDiff.myAssigned || col2 != myDiff.bestChangeAssigned)
//					diffCheck = -1.0f;
//				else
//					diffCheck = aijMatrix[row1][col2] + aijMatrix[row2][col1] - (aijMatrix[row1][col1] + aijMatrix[row2][col2]);
//			}
//			// We need to check that our previous calculation still holds
//			// It may not due to second order effects
//			if (diffCheck <= 0) {
//				if (myDiff.type == 0)
//					rowDifferences[myDiff.index] = emptyDiff;
//				else
//					columnDifferences[myDiff.index] = emptyDiff;
//				removeDifference(0);
//				continue;
//			}
//
//			// System.out.println("Happiness before switch:
//			// "+calculateTotalHappiness());
//			// So now we switch rows and columns
//			persons[row1] = col2;
//			if (row2 != -1) {
//				// if (col1 == -1)
//				// bannedSwitches[row1].add(row2);
//				persons[row2] = col1;
//			}
//			// if (col1 != -1)
//			objects[col1] = row2;
//			objects[col2] = row1;
//			// if (col1 == -1 && row2 == -1)
//			// return;
//
//			// System.out.println("Happiness after switch:
//			// "+calculateTotalHappiness());
//
//			// Now we update the modified rows and columns
//			switchedRows[0] = row1;
//			switchedRows[1] = row2;
//			switchedColumns[0] = col1;
//			switchedColumns[1] = col2;
//			int i;
//			for (i = 0; i < 2; i++) {
//				if (columnDifferences[switchedColumns[i]].index != -1) {
//					Difference toRemove = columnDifferences[switchedColumns[i]];
//					int z;
//					for (z = 1; z < numberOfObjects + numberOfPersons; z++) {
//						Difference toCheck= differences[z];
//						if (toCheck.index == -1)
//							break;
//						if (toCheck.index == toRemove.index && toCheck.type == toRemove.type) {
//							removeDifference(z);
//							break;
//						}
//					}
//					columnDifferences[switchedColumns[i]] = emptyDiff;
//				}
//				addColBestDifference(switchedColumns[i], 1);
//			}
//			for (i = 0; i < 2; i++) {
//				if (rowDifferences[switchedRows[i]].index != -1) {
//					Difference toRemove = rowDifferences[switchedRows[i]];
//					int z;
//					for (z = 1; z < numberOfObjects + numberOfPersons; z++) {
//						Difference toCheck= differences[z];
//						if (toCheck.index == -1)
//							break;
//						if (toCheck.index == toRemove.index && toCheck.type == toRemove.type) {
//							removeDifference(z);
//							break;
//						}
//					}
//					rowDifferences[switchedRows[i]] = emptyDiff;
//				}
//				addRowBestDifference(switchedRows[i], 1);
//			}
//			//		if (time(NULL) - start > initializationTimeLim) {
//			//			break;
//			//		}
//		}
//		// System.out.println("Total Happiness " +
//		// calculateTotalHappiness());
//		newTotalHappiness = calculateTotalHappiness();
//		if (newTotalHappiness == oldTotalHappiness) // || (SimulableSystem.currentTimeMillis() - start) > Conf.heuristicMaxTime
//			break;
//
//	}
//}

void removeDifference(int id) {
	differences[id] = emptyDiff;
	int i;
	for (i = id + 1; i < numberOfPersons + numberOfObjects; i++) {
		if (differences[i].value != -1) {
			differences[i - 1] = differences[i];
		} else {
			break;
		}
	}
}

void addColBestDifference(int colId, int sort) {
	if (colId == -1 || objects[colId] == -1)
		return;
	float maxDiff = 0.009;
	int bestChangeRow = -1;
	int myRow = objects[colId];
	int myCol = colId;
	int otherRow;
	for (otherRow = 0; otherRow < numberOfPersons; otherRow++) {
		int otherCol = persons[otherRow];
		if (otherCol == -1)
			continue;
		if (aijMatrix[otherRow][myCol] != negInf && aijMatrix[myRow][otherCol]
				!= negInf) {
			float difference = aijMatrix[myRow][otherCol]
					+ aijMatrix[otherRow][myCol] - (aijMatrix[myRow][myCol]
					+ aijMatrix[otherRow][otherCol]);
			if (difference > maxDiff) {
				maxDiff = difference;
				bestChangeRow = otherRow;
			}
		}
	}
	if (maxDiff > 0.1) {
		Difference myDifference;
		myDifference.index = colId;
		myDifference.bestChange = bestChangeRow;
		myDifference.type = 1;
		myDifference.value = maxDiff;

		myDifference.myAssigned = objects[colId];
		myDifference.bestChangeAssigned = persons[bestChangeRow];
		if (sort == 0) {
			differences[numberOfPersons + colId] = myDifference;
		} else {
			addSortedDifference(myDifference);
		}
		columnDifferences[colId] = myDifference;
	}
}

void addSortedDifferenceMod(Difference *myDifference) {
	int i;
	float maxDiff = myDifference->value;
	for (i = 0; i < numberOfPersons + numberOfObjects; i++) {
		if (maxDiff > differences[i].value) {
			Difference currentDiff = *myDifference;
			int j;
			for (j = i; j < numberOfObjects + numberOfPersons; j++) {
				if (differences[j].value != -1) {
					Difference temp = differences[j];
					differences[j] = currentDiff;
					currentDiff = temp;
				} else {
					differences[j] = currentDiff;
					break;
				}
			}
			break;
		}
	}
}

void addSortedDifference(Difference myDifference) {
	int i;
	float maxDiff = myDifference.value;
	for (i = 0; i < numberOfPersons + numberOfObjects; i++) {
		if (maxDiff > differences[i].value) {
			Difference currentDiff = myDifference;
			if (currentDiff.index != -1) {
				int row1 = currentDiff.index; // index of row of the difference
				// cell in the row of difference
				int col1 = persons[row1];
				int col2 = currentDiff.bestChange; // index of column of the best
				int row2 = objects[col2]; // index of row of the chosen in the
				currentDiff.mRmC = aijMatrix[row1][col1];
				currentDiff.mRoC = aijMatrix[row1][col2];
				currentDiff.oRmC = aijMatrix[row2][col1];
				currentDiff.oRoC = aijMatrix[row2][col2];
			}
			int j;
			for (j = i; j < numberOfObjects + numberOfPersons; j++) {
				if (differences[j].value != -1) {
					Difference temp = differences[j];
					differences[j] = currentDiff;
					currentDiff = temp;
				} else {
					differences[j] = currentDiff;
					break;
				}
			}
			break;
		}
	}
}

//void addRowBestDifferenceMod(int rowId, int sort,Difference *myDifference) {
//	if (myDifference->index != -1) {
//		myDifference->myAssigned = persons[rowId];
//		myDifference->bestChangeAssigned = objects[myDifference->bestChange];
//		if (sort == 0) {
//			differences[rowId] = *myDifference;
//			printf("r%d %f",rowId,differences[rowId].value);
//		}
//		else {
//			addSortedDifferenceMod(myDifference);
//		}
//		rowDifferences[rowId] = *myDifference;
//	}
//}

void addRowBestDifference(int rowId, int sort) {
	Difference myDifference = getRowBestDifference(rowId);
	if (myDifference.index != -1) {
		myDifference.myAssigned = persons[rowId];
		myDifference.bestChangeAssigned = objects[myDifference.bestChange];
		if (sort == 0) {
			differences[rowId] = myDifference;
		} else {
			addSortedDifference(myDifference);
		}
		rowDifferences[rowId] = myDifference;
	}
}

// Variable
Difference getRowBestDifference(int rowId) {
	if (rowId == -1)
		return emptyDiff;
	float maxDiff = 0.009f;
	int bestChangeCol = -1;
	int myCol = persons[rowId];
	if (myCol == -1)
		maxDiff = negInf;
	int myRow = rowId;
	int foundFreeObject = 0;
	int otherCol;
	for (otherCol = 0; otherCol < numberOfObjects; otherCol++) {//aijMatrix[rowId].keySet()
		int otherRow = objects[otherCol];
		float difference = negInf;
		// Person is not assigned
		if (myCol == -1) {
			// Object considered not assigned
			if (otherRow == -1) {
				// happiness value for the per-obj association
				difference = aijMatrix[myRow][otherCol];
				if (foundFreeObject == 0) {
					maxDiff = difference;
					bestChangeCol = otherCol;
				}
				foundFreeObject = 1;
			} else if (foundFreeObject == 0 && !bannedSwitches[myRow
					* numberOfObjects + otherRow])
				// object is not free
				// Compare old case with new case
				// pos...better me
				// neg...better him
				difference = aijMatrix[myRow][otherCol]
						- aijMatrix[otherRow][otherCol];
		} else if (otherRow == -1)
			// Compare old case with new case
			difference = aijMatrix[myRow][otherCol] - aijMatrix[myRow][myCol];
		else if (aijMatrix[otherRow][myCol] != negInf) {
			// Both assigned
			// Switch improves overall happiness of the two assignments
			difference = aijMatrix[myRow][otherCol]
					+ aijMatrix[otherRow][myCol] - (aijMatrix[myRow][myCol]
					+ aijMatrix[otherRow][otherCol]);
		}
		if (difference > maxDiff) {
			maxDiff = difference;
			bestChangeCol = otherCol;
		}
	}

	if (maxDiff < 0)
		maxDiff = -maxDiff;
	if (maxDiff > 0.1 || myCol == -1) {
		if (bestChangeCol == -1) {
			printf("Return");
			// Didn't find any worth switches because of the banning
			if (clearedBannedSwitches[myRow]) {
				persons[myRow] = -1;
				return emptyDiff;
			}
			clearedBannedSwitches[myRow] = 1;
			int x;
			for (x = 0; x < numberOfPersons; x++)
				bannedSwitches[myRow * numberOfObjects + x] = 0;
			return getRowBestDifference(rowId);
		}
		if (myCol == -1)
			maxDiff = maxDiff * 1000;
		Difference myDifference;
		myDifference.index = rowId;
		myDifference.bestChange = bestChangeCol;
		myDifference.type = 0;
		myDifference.value = maxDiff;
		myDifference.mRmC = 0.0;
		myDifference.mRoC = 0.0;
		myDifference.oRmC = 0.0;
		myDifference.oRoC = 0.0;
		return myDifference;
	}
	return emptyDiff;
}
