 /*
 *  Global.h
 *  heuristic CUDA
 *
 *  Created by Roberto Roverso on 25/08/09.
 *  Copyright 2009 Peerialism. All rights reserved.
 *
 */

#include "Difference.c"
#include "AijMatrix.c"

// Constant
extern float negInf;
extern int initializationTimeLim; // time lim for the initialization in seconds ...

// Variables
extern float** aijMatrix;
extern char* bannedSwitches;
extern int* persons;
extern int* objects;
extern char* clearedBannedSwitches;
extern Difference* differences;
extern Difference* columnDifferences;
extern Difference* rowDifferences;
extern Difference emptyDiff;
extern int numberOfPersons, numberOfObjects;

//#define MAC
#define CUDPP
//#define KERNEL_DEBUG
//#define USE_SHARED_DIFF_KERNEL
