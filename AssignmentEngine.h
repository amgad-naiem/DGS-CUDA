/*
 *  AssignmentEngine.h
 *  heuristic CUDA
 *
 *  Created by Roberto Roverso on 25/08/09.
 *  Copyright 2009 Peerialism. All rights reserved.
 *
 */

float calculateTotalHappiness();
float calculateIterationHappiness(int* persons);
Difference getRowBestDifference(int);
void addColBestDifference(int, int);
void addRowBestDifference(int, int);
void addRowBestDifferenceMod(int,int,Difference*);
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
