/*
 *  AssignmentEngine.cu
 *  heuristic CUDA
 *
 *  Created by Roberto Roverso on 25/08/09.
 *  Copyright 2009 Peerialism. All rights reserved.
 *
 */
// includes, system
#include <stdio.h>
#include <assert.h>
//#include <jni.h>
#include "Global.h"
#ifdef MAC
#include "sys/malloc.h" // mac os x
#else
#include "malloc.h" // linux, windows
#endif
#include <stdlib.h>
#include <iostream>
#include <string>
using namespace std;

//CUDA imports
#include <cuda.h>
#include <cutil_inline.h>
#include <cuda_runtime_api.h>
#include <cutil.h>

#ifdef CUDPP
#include <cudpp/cudpp.h>
#endif

// Include C files
#include "AssignmentEngine.h"
#include "Generator.h"
// include kernels
#include "BestDiffKernelGlobalGPU.cu"
#include "BestDiffKernelSharedGPU.cu"
#include "SwitchingKernelConflicts.cu"
#include "SortKernel.cu"
#include "InitAssignmentKernel.cu"

#define BLOCK_SIZE 16
#define DEFAULT_MULTI 8

#define TRUE 48
#define FALSE 49

typedef struct {
	float happiness;
	float time;
	float memoryTimer;
	float assTime;
} TestResult;

float* h_srtRes;

int isFeasible();

// CUDA related
void checkCUDAError(const char *msg);
void listCudaDevice();

//// CUDA Kernels
__global__ void evaluateDiff(AijMatrix A, Difference* diffs, float* srtDiffs,
		int* persons, int* objects, char* bannedSwitches,
		char* clearedBannedSwitches);

__global__ void placeDifferencesByIndex(Difference* diffs, int* index, int max);

__global__ void findMax(Difference* diffs, float* srtDiffs, AijMatrix A,
		int* persons, int* objects, char* bannedSwitches,
		char* clearedBannedSwitches, float* resDiffs);

__global__ void switching(AijMatrix A, Difference* differences,
		bool* conflicts, int* persons, int* objects, char* changedRows,
		char* changedCols, float* srtDiffs, int m);

__global__ void
evaluateDiffShared(Difference* diffs, AijMatrix A, int* persons, int* objects,
		char* bannedSwitches, char* clearedBannedSwitches, float* resDiffs);

bool* checkConflicts(Difference* differencesInternal, int n);

// Host
TestResult runDGSTest(int, int);
void hostInit(float** aijMatrix, int numberOfPersons, int numberOfObjects);
void gpuInit(float** aijMatrix, int numberOfPersons, int numberOfObjects);
void gpuInit2();
void fail(string&);

TestResult runHeuristic(float** aijMatrix, int numberOfPersons,
		int numberOfObjects);

void smartInitialAssignment();
void smartInitialAssignmentGPU();
void enhanceBySwitching();
void evaluateDifferences();
void sortDifferencesGPU();

void gpuTerninate();

//Utility functions
void printH(float** aijMatrix, int numberOfPersons, int numberOfObjects);
void printG(float* aijMatrix, int numberOfPersons, int numberOfObjects);

// Constants
int initializationTimeLim = 10; // time lim for the initialization in seconds ...
float negInf = -9999;

// Variables on Host
float** aijMatrix;
int* persons;
int* objects;
char* bannedSwitches;
char* clearedBannedSwitches;
Difference* differences;
Difference* differences_temp;
Difference* columnDifferences;
Difference* rowDifferences;
Difference emptyDiff;
int numberOfPersons, numberOfObjects;
int* reset;
float* h_Diffs;
int* h_bestChanges;
int* h_index;

// Variables on GPU
unsigned int tAijMSizeB;
unsigned int tPersonsSizeB;
unsigned int tObjectSizeB;
unsigned int tbestChangesB;
unsigned int tDiffsSizeB;
unsigned int tSrtDiffs;
unsigned int tBannedSwitches;
unsigned int tClearedBannedSwitches;
unsigned int indexSize;
AijMatrix d_aijM;
int* d_pers;
int* d_objs;
Difference d_emptyDiff;
Difference* d_differences;
//Difference* d_columnDifferences;
//Difference* d_rowDifferences;
float* d_srtDiffs;
float* d_DiffResults;
char* d_bannedSwitches;
char* d_clearedBannedSwitches;
bool* d_rowConflicts;
bool* d_colConflicts;
int* d_index;
bool* changedRows;
bool* changedCols;
bool* d_conflicts;

int pBoolSize;
int oBoolSize;
int conflictsS;

unsigned int freeMemDevice, totalMemDevice;

#ifdef CUDPP
CUDPPHandle scanplan = 0;
#endif

// Run option flags
bool useGenerator = false;
bool runCpu = false;
bool runGpu = true;
bool assignmentGpu = false;
bool assignmentCpu = true;
bool assignmentOnly = false;
bool niOut = false;
bool pOut = false;
bool pResInit = false;
bool pResAss = false;
bool pDbg = false;
bool pTimer = false;
bool mTests = false;
bool sGPU = false;
bool sortP = false;
bool sdk = false;
bool swg = false;

// Timers for benchmarking
float tEvaluateDiff = 0.0f;
float tSorting = 0.0f;
float tSwitching = 0.0f;
float tMemory = 0.0f;
int iterations = 0;
int seed = 7;

unsigned int timerProc = 0;
unsigned int timerSort = 0;
unsigned int timerED = 0;

int minMult = 0;
int blockSize = BLOCK_SIZE;

int maxMult = 10;

int n = 40;

int multi = DEFAULT_MULTI;

static unsigned long inKB(unsigned long bytes) {
	return bytes / 1024;
}

static unsigned long inMB(unsigned long bytes) {
	return bytes / (1024 * 1024);
}

static void printStats(CUdevice dev, unsigned long free, unsigned long total) {
#if CUDART_VERSION < 2020
#error "This CUDART version does not support mapped memory!\n"
#endif
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("Chosen GPU Device %d: \"%s\"\n", dev, deviceProp.name);
	printf("  Number of multiprocessors:                     %d\n",
			deviceProp.multiProcessorCount);
	printf("  Number of cores:                               %d\n", 8
			* deviceProp.multiProcessorCount);
	printf("  Clock rate:                                    %.2f GHz\n",
			deviceProp.clockRate * 1e-6f);
	printf("  Can Map Host Memory: 				 %s \n",
			(deviceProp.canMapHostMemory) ? "true" : "false");
	printf("  Free Mem: 				         %lu bytes (%lu KB) (%lu MB)\n", free,
			inKB(free), inMB(free));
	printf("  Total Mem: 					 %lu bytes (%lu KB) (%lu MB)\n", total, inKB(
			total), inMB(total));

	if (!deviceProp.canMapHostMemory) {
		fprintf(stderr, "Device %d cannot map host memory!\n", 0);
		exit(EXIT_FAILURE);
	}

}

// Main
int main(int argc, char** argv) {
	for (int i = 1; i < argc; i++) {
		if (strcmp(argv[i], "-l") == 0) {
			listCudaDevice();
			return 0;
		}
		if (strcmp(argv[i], "-c") == 0) {
			runCpu = true;
			runGpu = false;

		}
		if (strcmp(argv[i], "-ni") == 0) {
			niOut = true;

		}
		if (strcmp(argv[i], "-t") == 0) {
			pTimer = true;
		}
		if (strcmp(argv[i], "-ag") == 0) {
			assignmentGpu = true;
			assignmentCpu = false;
		}
		if (strcmp(argv[i], "-cg") == 0) {
			runCpu = true;
		}

		if (strcmp(argv[i], "-b") == 0) {
			blockSize = atoi(argv[i + 1]);
		}

		if (strcmp(argv[i], "-ao") == 0) {
			assignmentOnly = true;
		}

		if (strcmp(argv[i], "-m") == 0) {
			multi = atoi(argv[i + 1]);
		}

		if (strcmp(argv[i], "-ns") == 0) {
			n = atoi(argv[i + 1]);
		}

		if (strcmp(argv[i], "-p") == 0) {
			pOut = true;
		}
		if (strcmp(argv[i], "-ri") == 0) {
			pResInit = true;
		}

		if (strcmp(argv[i], "-swg") == 0) {
			swg = true;
		}
		if (strcmp(argv[i], "-ra") == 0) {
			pResAss = true;
		}
		if (strcmp(argv[i], "-d") == 0) {
			pDbg = true;
		}
		if (strcmp(argv[i], "-gen") == 0) {
			useGenerator = true;
		}
		if (strcmp(argv[i], "-sg") == 0) {
			sGPU = true;
		}
		if (strcmp(argv[i], "-so") == 0) {
			sortP = true;
		}
		if (strcmp(argv[i], "-sdk") == 0) {
			sdk = true;
		}

		if (strcmp(argv[i], "-seed") == 0) {
			seed = atoi(argv[i + 1]);
		}

		if (strcmp(argv[i], "-mT") == 0) {
			mTests = true;
			minMult = atoi(argv[i + 1]);
			maxMult = atoi(argv[i + 2]);
		}

	}

	int GPU_N;
	cutilSafeCall(cudaGetDeviceCount(&GPU_N));

	if (!niOut) {
		printf("CUDA-capable device count: %i\n", GPU_N);
	}
	for (int i = 0; i < GPU_N; i++) {
		CUdevice device;
		cuDeviceGet(&device, i);
		CUcontext ctx;
		cuCtxCreate(&ctx, 0, device);
		CUresult res = cuMemGetInfo(&freeMemDevice, &totalMemDevice);
		if (!niOut) {
			printStats(i, freeMemDevice, totalMemDevice);
		}
	}

	/*
	 * Check memory available
	 */
	numberOfPersons = blockSize * multi;
	numberOfObjects = blockSize * multi;
	TestResult r = runDGSTest(numberOfPersons, numberOfObjects);
	if (!assignmentOnly)
		printf("%d, %f, %f, %f, %f, %f, %f, %d, %d,%d,%f\n", numberOfPersons,
				r.happiness, r.time, tEvaluateDiff, tSorting, tSwitching,
				tMemory, iterations /*(r.memoryTimer)*/, isFeasible(), seed,
				r.assTime);
	else
		printf("%d, %d,%f\n", numberOfPersons, seed, r.assTime);
	return 0;
}

TestResult runDGSTest(int numberOfPersons, int numberOfObjects) {
	if (useGenerator) {
		if (!niOut) {
			printf("Using Euclidean Generator\n");
		}
		aijMatrix = genMatrix(numberOfPersons, numberOfObjects, seed);
	} else {
		if (!niOut) {
			printf("Using Random Generator\n");
		}
		// For testing purpose only
		int C = 3000;
		aijMatrix = (float **) malloc(numberOfPersons * sizeof(float *));

		float *aijPtr = (float *) malloc(numberOfPersons * numberOfObjects
				* sizeof(float));
		for (int i = 0; i < numberOfPersons; i++) {
			aijMatrix[i] = aijPtr + (i * numberOfObjects);
		}

		for (int i = 0; i < numberOfPersons; i++) {
			for (int j = 0; j < numberOfObjects; j++) {
				aijMatrix[i][j] = random() % C;
			}
		}

	}

	//RUN
	TestResult r = runHeuristic(aijMatrix, numberOfPersons, numberOfObjects);

	//	printf("Correct %d\n ", isFeasible());
	return r;
}

/**
 * Initialize structure on host memory
 */
void hostInit(float** aijMatrix, int numberOfPersons, int numberOfObjects) {

	emptyDiff.index = -1;
	emptyDiff.myAssigned = -1;
	emptyDiff.bestChangeAssigned = -1;

	bannedSwitches = (char *) malloc(numberOfPersons * numberOfPersons
			* sizeof(bool));

	int i, j;
	clearedBannedSwitches = (char *) malloc(numberOfPersons * sizeof(char));
	persons = (int *) malloc(numberOfPersons * sizeof(int));
	differences = (Difference *) malloc((numberOfPersons + numberOfObjects)
			* sizeof(Difference));
	rowDifferences
			= (Difference *) malloc(numberOfPersons * sizeof(Difference));
	for (i = 0; i < numberOfPersons; i++) {
		clearedBannedSwitches[i] = 0;
		persons[i] = -1;
		for (j = 0; j < numberOfPersons; j++)
			bannedSwitches[i * numberOfObjects + j] = 0;
		rowDifferences[i] = emptyDiff;
		differences[i] = emptyDiff;
	}

	columnDifferences = (Difference *) malloc(numberOfObjects
			* sizeof(Difference));
	objects = (int *) malloc(numberOfObjects * sizeof(int));
	for (i = 0; i < numberOfObjects; i++) {
		objects[i] = -1;
		columnDifferences[i] = emptyDiff;
		differences[numberOfPersons + i] = emptyDiff;
	}

	//	printH(aijMatrix, numberOfPersons, numberOfObjects);
}

/**
 * Initialize structure on video memory and upload
 */
void gpuInit(float** aijMatrix, int numberOfPersons, int numberOfObjects) {

	// GPU Memory management
	// allocate host memory

	// allocate device memory
	tPersonsSizeB = sizeof(int) * numberOfPersons;
	tObjectSizeB = sizeof(int) * numberOfObjects;
	tSrtDiffs = sizeof(float) * (numberOfObjects + numberOfPersons);
	// Best Changes are included in Diff struct
	//	tbestChangesB = sizeof(int) * (numberOfPersons + numberOfObjects);
	tDiffsSizeB = sizeof(Difference) * (numberOfPersons + numberOfObjects);
	//	unsigned int tRowDiffsSizeB = sizeof(Difference) * (numberOfPersons);
	//	unsigned int tColDiffsSizeB = sizeof(Difference) * (numberOfObjects);
	//	int resetB = sizeof(bool);
	tClearedBannedSwitches = sizeof(char) * numberOfPersons;

	unsigned int tAijMSize = numberOfPersons * numberOfObjects;
	tAijMSizeB = sizeof(float) * tAijMSize;
	tBannedSwitches = sizeof(char) * tAijMSize;

	int totalBonGpu = tPersonsSizeB + tObjectSizeB + tAijMSizeB + tDiffsSizeB
			+ tBannedSwitches + tClearedBannedSwitches;

	if (!niOut) {
		printf("Memory used on GPU: %d Bytes\n", totalBonGpu);
	}

	if (totalBonGpu > freeMemDevice) {
		printf("Warning: not enough memory available on GPU: %d Bytes\n",
				freeMemDevice);
	}

	float* h_aijM = (float*) malloc(tAijMSizeB);

	for (int i = 0; i < numberOfPersons; i++) {
		for (int j = 0; j < numberOfObjects; j++) {
			h_aijM[i * numberOfPersons + j] = aijMatrix[i][j];
			//			printf("%f ", h_aijM[i * numberOfPersons + j]);
		}
		//		printf("\n ");
	}

	d_aijM.height = numberOfPersons;
	d_aijM.width = numberOfObjects;

	//	// Init all the diffs to null before uploading
	//	h_Diffs = (float *) malloc(tDiffsSizeB);
	//	h_bestChanges = (int*) malloc(tbestChangesB);
	//
	//	for (int k = 0; k < numberOfPersons + numberOfObjects; k++)
	//	{
	//		//		assigned[k] = false;
	//		h_Diffs[k] = negInf;
	//		h_bestChanges[k] = -1;
	//	}

	//aijMatrix
	cutilSafeCall(cudaMalloc((void**) &d_aijM.els, tAijMSizeB));
	//persons
	cutilSafeCall(cudaMalloc((void**) &d_pers, tPersonsSizeB));
	//object
	cutilSafeCall(cudaMalloc((void**) &d_objs, tObjectSizeB));
	// Reset flag
	//	cutilSafeCall(cudaMalloc((void**) &d_reset, resetB));
	// Banned Switches
	cutilSafeCall(cudaMalloc((void**) &d_bannedSwitches, tBannedSwitches));
	// Cleared Banned Switches
	cutilSafeCall(cudaMalloc((void**) &d_clearedBannedSwitches,
			tClearedBannedSwitches));
	// BlockSize
	//	cutilSafeCall(cudaMalloc((void**) &d_blockSize, sizeof(int)));

	// copy host memory to device
	cutilSafeCall(cudaMemcpy(d_aijM.els, h_aijM, tAijMSizeB,
			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_pers, persons, tPersonsSizeB,
			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_objs, objects, tObjectSizeB,
			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_bannedSwitches, bannedSwitches, tBannedSwitches,
			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_clearedBannedSwitches, clearedBannedSwitches,
			tClearedBannedSwitches, cudaMemcpyHostToDevice));

	//temporary diffs for sorting on GPU
	cutilSafeCall(cudaMalloc((void**) &d_srtDiffs, tSrtDiffs));
	//	int blockS[1];
	//	blockS[0] = blockSize;
	//	cutilSafeCall(cudaMemcpy(d_blockSize, blockS, sizeof(int),
	//			cudaMemcpyHostToDevice));

	//	reset = (int *) malloc(sizeof(int));
	//	cutilSafeCall(cudaMemcpy(d_reset, reset, resetB, cudaMemcpyHostToDevice));

	// allocate device memory for result (diff values)
	cutilSafeCall(cudaMalloc((void**) &d_differences, tDiffsSizeB));
	// allocate device memory for result (bestchange values)
	//	cutilSafeCall(cudaMalloc((void**) &d_bestChanges, tbestChangesB));

	//	cutilSafeCall(cudaMalloc((void**) &d_rowDifferences, tRowDiffsSizeB));
	//	cutilSafeCall(cudaMalloc((void**) &d_columnDifferences, tColDiffsSizeB));

	cutilSafeCall(cudaMemcpy(d_differences, differences, tDiffsSizeB,
			cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_rowDifferences, rowDifferences, tRowDiffsSizeB,
	//			cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_columnDifferences, columnDifferences,
	//			tColDiffsSizeB, cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_bestChanges, h_bestChanges, tbestChangesB,
	//			cudaMemcpyHostToDevice));

	if (sdk) {
		//ALLOCATE ROOM FOR THE RESULTING DIFFERENCES
		cutilSafeCall(cudaMalloc((void**) &d_DiffResults, tAijMSizeB));
	}

	cuMemGetInfo(&freeMemDevice, &totalMemDevice);

	cutilCheckError(cutCreateTimer(&timerED));
	cutilCheckError(cutCreateTimer(&timerSort));
	cutilCheckError(cutCreateTimer(&timerProc));

	n = numberOfPersons;

	int pBoolSize = numberOfPersons * sizeof(bool);
	int oBoolSize = numberOfObjects * sizeof(bool);
	int conflictsS = n * sizeof(bool);

	changedRows = (bool *) malloc(pBoolSize);
	changedCols = (bool *) malloc(oBoolSize);

	cutilSafeCall(cudaMalloc((void**) &d_rowConflicts, pBoolSize));
	cutilSafeCall(cudaMalloc((void**) &d_colConflicts, oBoolSize));

	cutilSafeCall(cudaMalloc((void**) &d_conflicts, conflictsS));

	if (!niOut)
		printf("Memory after allocation: %f%% free, %f%% used\n", 100.0
				* freeMemDevice / (double) totalMemDevice, 100.0
				* (totalMemDevice - freeMemDevice) / (double) totalMemDevice);
}

void gpuTerninate() {
	cutilSafeCall(cudaFree(d_aijM.els));
	cutilSafeCall(cudaFree(d_pers));
	cutilSafeCall(cudaFree(d_objs));
	cutilSafeCall(cudaFree(d_differences));
	cutilSafeCall(cudaFree(d_DiffResults));
//	cutilSafeCall(cudaFree(d_bestChanges));
	cutilSafeCall(cudaFree(d_bannedSwitches));
	cutilSafeCall(cudaFree(d_clearedBannedSwitches));
	cutilSafeCall(cudaFree(d_srtDiffs));
}

unsigned int memoryTimer = 0;

TestResult runHeuristic(float** aijMatrix, int numberOfPersons,
		int numberOfObjects) {

	//Init
	hostInit(aijMatrix, numberOfPersons, numberOfObjects);

	//	if (len > 0)
	//		smartInitialAssignmentWithInitial( initialAssignment);
	//	else
	//		smartInitialAssignment();
	//	enhanceBySwitching();
	if (!niOut) {
		printf("Entities %d\n", numberOfPersons);
		printf("Block Size %d\n", blockSize);
	}

	unsigned int timer = 0;
	cutilCheckError(cutCreateTimer(&timer));
	cutilCheckError(cutCreateTimer(&memoryTimer));
	//	if (assignmentGpu)
	//	{
	//		//		gpuInit(aijMatrix, numberOfPersons, numberOfObjects);
	//		if (!niOut)
	//		{
	//			printf("-----GPU Assignment------\n");
	//		}
	//		smartInitialAssignmentGPU();
	//	}

	TestResult r;
	if (assignmentCpu) {
		cutilCheckError(cutCreateTimer(&timer));
		cutilCheckError(cutStartTimer(timer));
		if (!niOut) {

			printf("-----CPU Assignment------\n");
		}
		smartInitialAssignment();

		cutilCheckError(cutStopTimer(timer));
		r.assTime = cutGetTimerValue(timer);
		if (!niOut) {
			printf("Done Assignment\n");
			printf("Processing time: %f (ms)\n", r.assTime);
		}
	} else if (assignmentGpu) {
		gpuInit(aijMatrix, numberOfPersons, numberOfObjects);
		cutilCheckError(cutCreateTimer(&timer));
		cutilCheckError(cutStartTimer(timer));
		if (!niOut) {

			printf("-----GPU Assignment------\n");
		}
		smartInitialAssignment();

		cutilCheckError(cutStopTimer(timer));
		r.assTime = cutGetTimerValue(timer);
		if (!niOut) {
			printf("Done Assignment\n");
			printf("Processing time: %f (ms)\n", r.assTime);
		}
	}

	if (pResInit) {
		for (int i = 0; i < numberOfPersons; i++) {
			printf("aij ");
			printf("%d ", i);
			printf("%d ", persons[i]);
			printf("%f \n", aijMatrix[i][persons[i]]);
		}

	}

	if (!assignmentOnly) {

		if (!niOut) {
			printf("Enhance By Switching ");
		}
		if (runGpu) {
			if (!niOut) {
				printf("on GPU\n");
			}
			if (!assignmentGpu) {
				gpuInit(aijMatrix, numberOfPersons, numberOfObjects);
			}
		} else {
			if (!niOut) {
				printf("on CPU\n");
			}
		}

		cutilCheckError(cutResetTimer(timer));
		cutilCheckError(cutStartTimer(timer));
		// Enhance by switching
		enhanceBySwitching();
		cutilCheckError(cutStopTimer(timer));

		float happiness = calculateTotalHappiness();

		float tValue = cutGetTimerValue(timer);
		if (pResAss) {
			for (int i = 0; i < numberOfPersons; i++) {
				printf("aij ");
				printf("%d ", i);
				printf("%d ", persons[i]);
				printf("%f \n", aijMatrix[i][persons[i]]);
			}

		}
		r.time = tValue;
		r.happiness = happiness;
		//		float v = cutGetTimerValue(memoryTimer);
		//		r.memoryTimer = v;
	}

	if (runGpu || assignmentGpu) {
		if (!niOut && !assignmentOnly) {
			printf("Cleaning GPU state\n");
		}
		cutilSafeCall(cudaMemcpy(persons, d_pers, tPersonsSizeB,
				cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(objects, d_objs, tObjectSizeB,
				cudaMemcpyDeviceToHost));
		gpuTerninate();
	}

	return r;

}

int isFeasible() {
	int* numAssignedPersons = (int *) malloc(numberOfPersons * sizeof(int));
	int* numAssignedObjects = (int *) malloc(numberOfObjects * sizeof(int));
	int i;
	for (i = 0; i < numberOfPersons; i++)
		numAssignedPersons[i] = 0;
	for (i = 0; i < numberOfObjects; i++)
		numAssignedObjects[i] = 0;
	for (i = 0; i < numberOfPersons; i++) {
		int assignedObject = persons[i];
		numAssignedObjects[assignedObject] = numAssignedObjects[assignedObject]
				+ 1;
	}
	for (i = 0; i < numberOfObjects; i++) {
		int assignedPerson = objects[i];
		numAssignedPersons[assignedPerson] = numAssignedPersons[assignedPerson]
				+ 1;
	}

	for (i = 0; i < numberOfPersons; i++)
		if (numAssignedPersons[i] > 1)
			return 0;
	for (i = 0; i < numberOfObjects; i++)
		if (numAssignedObjects[i] > 1)
			return 0;
	return 1;
}

void gpuInit2() {

	int size = numberOfPersons + numberOfObjects;
	h_srtRes = (float*) malloc(sizeof(float) * size);

	for (int var = 0; var < size; ++var) {
		h_srtRes[var] = 0.0f;
	}

	cutilSafeCall(cudaMemcpy(d_srtDiffs, h_srtRes, sizeof(float) * size,
			cudaMemcpyHostToDevice));

#ifdef CUDPP

	if (sGPU) {
		int size = numberOfPersons + numberOfObjects;
		CUDPPConfiguration config;
		config.algorithm = CUDPP_SORT_RADIX;
		config.datatype = CUDPP_FLOAT;
		config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

		CUDPPResult result = cudppPlan(&scanplan, config, size, 1, 0);

		if (CUDPP_SUCCESS != result) {
			string eMsg = "Error creating CUDPPPlan";
			fail(eMsg);
		}
		indexSize = (size) * sizeof(int);
		h_index = (int*) malloc(indexSize);
		cutilSafeCall(cudaMalloc((void**) &d_index, indexSize));

	}
#endif
}

void evaluateDifferences() {

	if (runGpu) {
		if (pDbg) {
			printf("Eval Diff Phase Start\n");
		}

		dim3 threads;
		dim3 grid;

		if (!sdk) {
			threads.x = blockSize;
			threads.y = 1;
			int s = numberOfPersons / blockSize + (numberOfPersons % blockSize
					== 0 ? 0 : 1);
			grid.x = s;
			grid.y = 1;
		} else {
			threads.x = blockSize;
			threads.y = blockSize;
			grid.x = numberOfObjects / threads.x;
			grid.y = numberOfPersons / threads.y;
		}
		if (pDbg) {
			printf("Pre-MemCpy Over\n");
		}
//		unsigned int timerEDSub = 0;
//		cutilCheckError(cutCreateTimer(&timerEDSub));
//		unsigned int timerEDSubb = 0;
//		cutilCheckError(cutCreateTimer(&timerEDSubb));

		if (sdk) {

			if (!niOut)
				printf("GRID %d %d TH %d %d \n", grid.y, grid.x, threads.x,
						threads.y);
			evaluateDiffShared <<<grid,threads>>>(d_differences, d_aijM, d_pers, d_objs,d_bannedSwitches,d_clearedBannedSwitches,d_DiffResults);
			CUT_CHECK_ERROR("bestDiff");
			//			cutilCheckError(cutStopTimer(timerEDSub));
			//			if (pTimer)unsigned int timerEDSub = 0;
			//		cutilCheckError(cutCreateTimer(&timerEDSub));
			//		unsigned int timerEDSubb = 0;
			//		cutilCheckError(cutCreateTimer(&timerEDSubb));
			//				printf("EDSub time: %f (ms)\n", cutGetTimerValue(timerEDSub));
			//			cutilCheckError(cutStartTimer(timerEDSubb));
			threads.x = blockSize;
			threads.y = 1;
			int s = numberOfPersons / blockSize + (numberOfPersons % blockSize
					== 0 ? 0 : 1);
			grid.x = s;
			grid.y = 1;
findMax		<<<grid,threads>>>(d_differences,d_srtDiffs, d_aijM, d_pers, d_objs,d_bannedSwitches,d_clearedBannedSwitches,d_DiffResults);
		//			cutilCheckError(cutStopTimer(timerED));
		//			if (pTimer)
		//				printf("EDSubb time: %f (ms)\n", cutGetTimerValue(timerEDSubb));
	} else
	{
		if (!niOut)
		printf("GRID %d %d TH %d %d \n", grid.y, grid.x, threads.x,
				threads.y);
//		cutilCheckError(cutStartTimer(timerED));
		evaluateDiff<<<grid,threads>>>( d_aijM, d_differences,d_srtDiffs,d_pers, d_objs,d_bannedSwitches,d_clearedBannedSwitches);
//		cutilCheckError(cutStopTimer(timerED));
//		if (pTimer)
//		printf("EDSub time: %f (ms)\n", cutGetTimerValue(timerEDSub));
		//			cutilCheckError(cutStartTimer(timerEDSubb));
		CUT_CHECK_ERROR("bestDiff");
	}
	cudaThreadSynchronize();

	if (pDbg)
	{
		printf("Post-MemCpy Over\n");
	}
} else
{
	for (int l = 0; l < numberOfPersons; l++)
	{
		// Find the best objective improvement
		addRowBestDifference(l, 0);
	}
}

}

void sortDifferencesGPU() {
#ifdef CUDPP
	int size = numberOfPersons + numberOfObjects;

	for (int var = 0; var < numberOfPersons + numberOfObjects; ++var) {
		h_index[var] = var;

	}

	if (pDbg) {
		printf("Pre-sort\n");
	}
	cutilCheckError(cutStartTimer(timerProc));
	cutilSafeCall(cudaMemcpy(d_index, h_index, indexSize,
			cudaMemcpyHostToDevice));
	cutilCheckError(cutStopTimer(timerProc));

	CUDPPResult res = cudppSort(scanplan, d_srtDiffs, d_index, 32, size);

	if (CUDPP_SUCCESS != res) {
		string eMsg = "Error sorting on GPU\n";
		fail(eMsg);
	}
	CUT_CHECK_ERROR("cudppRadixSort");

	if (pDbg) {
		printf("Post-sort\n");
	}

	dim3 threads;
	dim3 grid;

	threads.x = blockSize;
	threads.y = 1;
	int s = (numberOfPersons + numberOfObjects) / blockSize + ((numberOfPersons
			+ numberOfObjects) % blockSize == 0 ? 0 : 1);
	grid.x = s;
	grid.y = 1;

	placeDifferencesByIndex<<<grid,threads>>>(d_differences,d_index,size);

	if (pDbg) {
		printf("Post-red\n");
	}
#endif
}

void fail(string& s) {
	cout << s;
	exit(-1);
}

void smartInitialAssignmentGPU() {
	evaluateDifferences();
}

unsigned int timerSwitching = 0;

void enhanceBySwitching() {
	cutilCheckError(cutCreateTimer(&timerSwitching));

	float newTotalHappiness, oldTotalHappiness;
	// int counter = 1;
	//  time_t start = time(NULL);

	if (sGPU) {
		if (!niOut)
			printf("Sorting on GPU\n");
		gpuInit2();
	} else {
		if (!niOut)
			printf("Sorting on CPU \n");
	}

	int count = 0;
	int countInternal = 0;
	while (1) {
		if (!niOut) {
			printf("\nIteration %d\n", count++);
		}
		iterations++;

		oldTotalHappiness = calculateTotalHappiness();

//		cutilCheckError(cutStartTimer(timerED));
		//--------
		evaluateDifferences();
		return;
		//--------
//		cutilCheckError(cutStopTimer(timerED));
//		if (pTimer)
//			printf("ED time: %f (ms)\n", cutGetTimerValue(timerED));

		tEvaluateDiff += cutGetTimerValue(timerED);
		cutilCheckError(cutResetTimer(timerED));

		cutilCheckError(cutStartTimer(timerSort));
		if (sGPU) {
			//			if (sortP)
			//			{
			//				cutilSafeCall(cudaMemcpy(differences, d_differences,
			//						tDiffsSizeB, cudaMemcpyDeviceToHost));
			//				cutilSafeCall(cudaMemcpy(h_srtRes, d_srtDiffs, sizeof(float)
			//						* (numberOfObjects + numberOfPersons),
			//						cudaMemcpyDeviceToHost));
			//				printf("before sorting\n");
			//				int count2 = 0;
			//				for (int var = 0; var < numberOfPersons + numberOfObjects; ++var)
			//				{
			//					count2++;
			//					printf("\t %d i%d->%f srt %f", var, differences[var].index,
			//							differences[var].value, h_srtRes[var]);
			//					if (count2 == 8)
			//					{
			//						printf("\n");
			//						count2 = 0;
			//					}
			//				}
			//			}

			sortDifferencesGPU();
			//			if (!swg)
			cutilCheckError(cutStartTimer(timerProc));
			cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
					cudaMemcpyDeviceToHost));
			cutilCheckError(cutStopTimer(timerProc));
		} else {

			cutilCheckError(cutStartTimer(timerProc));

			cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
					cudaMemcpyDeviceToHost));
			cutilCheckError(cutStopTimer(timerProc));
			if (sortP) {
				printf("before sorting\n");
				int count2 = 0;
				for (int var = 0; var < numberOfPersons + numberOfObjects; ++var) {
					count2++;
					printf("\t %d i%d->%f", var, differences[var].index,
							differences[var].value);
					if (count2 == 8) {
						printf("\n");
						count2 = 0;
					}
				}
			}

			qsort(differences, numberOfPersons + numberOfPersons,
					sizeof(Difference), compare_differences);
			if (swg) {
				cutilCheckError(cutStartTimer(timerProc));
				cutilSafeCall(cudaMemcpy(d_differences, differences,
						tDiffsSizeB, cudaMemcpyHostToDevice));
				cutilCheckError(cutStopTimer(timerProc));
			}
		}
		cutilCheckError(cutStopTimer(timerSort));

		float ts = cutGetTimerValue(timerSort);
		tSorting += ts;

		cutilCheckError(cutResetTimer(timerSort));
		if (pTimer)
			printf("Sort time: %f (ms)\n", ts);

		if (sortP) {
			printf("After sorting\n");
			int count2 = 0;
			for (int var = 0; var < numberOfPersons + numberOfObjects; ++var) {
				count2++;
				printf("\t %d i%d->%f", var, differences[var].index,
						differences[var].value);
				if (count2 == 8) {
					printf("\n");
					count2 = 0;
				}
			}
		}
		if (pDbg)
			printf("Before Inner loop\n");

		if (pDbg)
			printf("Inner loop\n");

		//		cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
		//				cudaMemcpyDeviceToHost));

		cutilCheckError(cutStartTimer(timerSwitching));
		while (differences[0].index > 1) {
			countInternal++;
			if (countInternal % 20 == 0)
				printf("internal iterations: %d", countInternal);
			int i;
			for (i = 0; i < numberOfPersons; i++)
				changedRows[i] = false;
			for (i = 0; i < numberOfObjects; i++)
				changedCols[i] = false;

			// check diffs here

			int md = 0;
			for (int var = 0; var < numberOfPersons + numberOfObjects; ++var) {
				if (differences[var].index != -1)
					md++;
				else
					break;
			}

			bool* conflicts = checkConflicts(differences, md);

			int m = 0;
			for (i = 0; i < n; i++)
				if (!conflicts[i])
					m++;
			if (swg) {
				if (pDbg)
					printf("parallel part start\n");
				cutilCheckError(cutStartTimer(timerProc));
				cutilSafeCall(cudaMemcpy(d_rowConflicts, changedRows,
						pBoolSize, cudaMemcpyHostToDevice));
				cutilSafeCall(cudaMemcpy(d_colConflicts, changedCols,
						oBoolSize, cudaMemcpyHostToDevice));

				cutilSafeCall(cudaMemcpy(d_conflicts, conflicts, sizeof(bool)
						* n, cudaMemcpyHostToDevice));
				//				cutilSafeCall(cudaMemcpy(d_rowConflicts, changedRows,
				//						pBoolSize, cudaMemcpyHostToDevice));
				//				cutilSafeCall(cudaMemcpy(d_colConflicts, changedCols,
				//						oBoolSize, cudaMemcpyHostToDevice));
				cutilCheckError(cutStopTimer(timerProc));
				tSorting += cutGetTimerValue(timerProc);
				cutilCheckError(cutResetTimer(timerProc));

				dim3 threads;
				dim3 grid;

				threads.x = blockSize;
				threads.y = 1;
				int s = n / blockSize + (n % blockSize == 0 ? 0 : 1);
				if (s < 1)
					s = 1;

				grid.x = s;
				grid.y = 1;

				switching<<<grid,threads>>>(d_aijM, d_differences,d_conflicts, d_pers,d_objs,d_rowConflicts,d_colConflicts, d_srtDiffs, n);

				cutilCheckError(cutStartTimer(timerProc));
				cutilSafeCall(cudaMemcpy(changedRows, d_rowConflicts,
						pBoolSize, cudaMemcpyDeviceToHost));
				cutilSafeCall(cudaMemcpy(changedCols, d_colConflicts,
						oBoolSize, cudaMemcpyDeviceToHost));
				cutilSafeCall(cudaMemcpy(persons, d_pers, tPersonsSizeB,
						cudaMemcpyDeviceToHost));
				cutilSafeCall(cudaMemcpy(objects, d_objs, tObjectSizeB,
						cudaMemcpyDeviceToHost));

				cutilSafeCall(cudaMemcpy(differences, d_differences,
						tDiffsSizeB, cudaMemcpyDeviceToHost));
				cutilCheckError(cutStopTimer(timerProc));
				tSorting += cutGetTimerValue(timerProc);
				cutilCheckError(cutResetTimer(timerProc));
			} else {
				for (i = 0; i < n; i++) {
					Difference myDiff = differences[i];
					if (conflicts[i]) {
						if (myDiff.type == 0)
							changedRows[i] = true;
						else
							changedCols[i] = true;
					}
				}

				for (i = 0; i < n; i++) {
					if (!conflicts[i]) {

						Difference myDiff = differences[i];
						int row1, row2, col1, col2;
						// Here I need to retrieve the 2 columns and 2 rows that will be
						// altered due to switch ...
						// to not reUpdate all the differences in the Tree
						float diffCheck;
						if (myDiff.type == 0) { // Found in row..i.e. switching happens
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
									!= myDiff.bestChangeAssigned) {
								diffCheck = -1.0;
							} else if (row2 == -1) {
								diffCheck = aijMatrix[row1][col2]
										- aijMatrix[row1][col1];
							} else {
								diffCheck = aijMatrix[row1][col2]
										+ aijMatrix[row2][col1]
										- (aijMatrix[row1][col1]
												+ aijMatrix[row2][col2]);
							}
						} else {
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
								diffCheck = aijMatrix[row1][col2]
										+ aijMatrix[row2][col1]
										- (aijMatrix[row1][col1]
												+ aijMatrix[row2][col2]);
						}
						//printf("DiffCheck: \n");
						// We need to check that our previous calculation still holds
						// It may not due to second order effects
						if (diffCheck <= 0) {
							continue;
						}
						//
						//				// System.out.println("Happiness before switch:
						//				// "+calculateTotalHappiness());
						//				// So now we switch rows and columns
						persons[row1] = col2;
						if (row2 != -1) {
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
					}
				}

				//				cutilSafeCall(cudaMemcpy(d_pers, persons, tPersonsSizeB,
				//						cudaMemcpyHostToDevice));
				//				cutilSafeCall(cudaMemcpy(d_objs, objects, tObjectSizeB,
				//						cudaMemcpyHostToDevice));

			}
			//			cutilCheckError(cutStopTimer(timerSwitching));
			//
			//			if (pTimer)
			//				printf("M %d,PC time for switching first part: %f (ms)\n", m,
			//						cutGetTimerValue(timerSwitching));
			//
			//			cutilCheckError(cutStartTimer(timerSwitching));
			if (pDbg)
				printf("parallel part over\n");
			// remove first n differences
			for (i = 0; i < n; i++) {
				differences[i] = emptyDiff;
			}
			for (i = n; i < numberOfPersons + numberOfObjects; i++) {
				if (differences[i].value != -1) {
					differences[i - n] = differences[i];
				} else {
					break;
				}
			}

			if (pDbg)
				printf("re-eval row start\n");

			// re-evaluate diffs for rows
			for (i = 0; i < numberOfPersons; i++) {
				if (changedRows[i]) {
					// remove first before adding
					if (rowDifferences[i].index != -1) {
						Difference toRemove = rowDifferences[i];
						int z;
						for (z = 1; z < numberOfObjects + numberOfPersons; z++) {
							Difference toCheck = differences[z];
							if (toCheck.index == -1)
								break;
							if (toCheck.index == toRemove.index && toCheck.type
									== toRemove.type) {
								removeDifference(z);
								break;
							}
						}
						rowDifferences[i] = emptyDiff;
					}
					// add it .. can use gpu version for them all
					addRowBestDifference(i, 1);
				}
			}

			if (pDbg)
				printf("re-eval row end\n");

			if (pDbg)
				printf("re-eval col start\n");
			for (i = 0; i < numberOfObjects; i++) {
				if (changedCols[i]) {
					if (columnDifferences[i].index != -1) {
						Difference toRemove = columnDifferences[i];
						int z;
						for (z = 1; z < numberOfObjects + numberOfPersons; z++) {
							Difference toCheck = differences[z];
							if (toCheck.index == -1)
								break;
							if (toCheck.index == toRemove.index && toCheck.type
									== toRemove.type) {
								removeDifference(z);
								break;
							}
						}
						columnDifferences[i] = emptyDiff;
					}
					addColBestDifference(i, 1);
				}
			}
			if (pDbg)
				printf("re-eval col end\n");

			cutilCheckError(cutStartTimer(timerProc));
			//		if (time(NULL) - start > initializationTimeLim) {
			cutilSafeCall(cudaMemcpy(d_differences, differences, tDiffsSizeB,
					cudaMemcpyHostToDevice));
			cutilCheckError(cutStopTimer(timerProc));
			tSorting += cutGetTimerValue(timerProc);
			cutilCheckError(cutResetTimer(timerProc));

		}
		cutilCheckError(cutStopTimer(timerSwitching));
		tSwitching += cutGetTimerValue(timerSwitching);

		if (sortP) {
			printf("After all\n");
			int count2 = 0;
			for (int var = 0; var < numberOfPersons + numberOfObjects; ++var) {
				count2++;
				printf("\t %d i%d->%f", var, differences[var].index,
						differences[var].value);
				if (count2 == 8) {
					printf("\n");
					count2 = 0;
				}
			}
		}

		//		cutilCheckError(cutStopTimer(timerSwitching));
		//		tSwitching += cutGetTimerValue(timerSwitching);

		tMemory += cutGetTimerValue(timerProc);
		cutilCheckError(cutResetTimer(timerSwitching));

		if (pTimer)
			printf("PC time for switching second: %f (ms)\n", cutGetTimerValue(
					timerSwitching));

		//		cutilCheckError(cutResetTimer(timerSwitching));

		if (iterations > 1) {
			newTotalHappiness = calculateTotalHappiness();
			if (newTotalHappiness == oldTotalHappiness) {
				if (!niOut) {
					printf("Finished\n");
				}
				break;
			}
		}
	}
}

bool* checkConflicts(Difference* differencesInternal, int n) {
	// n is the first n differences to check
	// n < numberOfPersons + numberOfObjects
	int i;
	bool* changedRows = (bool *) malloc(numberOfPersons * sizeof(bool));
	for (i = 0; i < numberOfPersons; i++)
		changedRows[i] = false;
	bool* conflicts = (bool *) malloc(n * sizeof(bool));
	for (i = 0; i < n; i++)
		conflicts[i] = false;
	for (i = 0; i < n; i++) {
		Difference myDiff = differencesInternal[i];
		int row1, row2, col1, col2;
		if (myDiff.type == 0) {
			row1 = myDiff.index; // index of row of the difference
			col2 = myDiff.bestChange; // index of column of the best
			row2 = objects[col2]; // index of row of the chosen in the
		} else {
			col1 = myDiff.index; // index of column of the difference
			row1 = objects[col1]; // index of row of the chosen cell
			row2 = myDiff.bestChange; // index of row of the best cell
		}
		if (changedRows[row1] || changedRows[row2]) {
			conflicts[i] = true;
		} else {
			changedRows[row1] = true;
			changedRows[row2] = true;
		}
	}

	return conflicts;
}

void smartInitialAssignment() {
	int row1, curRow, col2, i;

	for (i = 0; i < numberOfPersons; i++) {

		curRow = i;
		while (curRow != -1) {
			Difference myDiff = getRowBestDifference(curRow);
			if (myDiff.index != -1) {
				row1 = myDiff.index; // index of row of the difference
				col2 = myDiff.bestChange; // index of column of the best
				// cell in the row of difference
				curRow = objects[col2]; // index of row of the chosen in the
				// column of the best cell in the
				// difference row
				persons[row1] = col2;
				objects[col2] = row1;
				if (curRow != -1) {
					persons[curRow] = -1;
					bannedSwitches[row1 * numberOfObjects + curRow] = 1;
				}
			}
		}
	}
}

void printH(float** aijMatrix, int numberOfPersons, int numberOfObjects) {
	for (int i = 0; i < numberOfPersons; i++) {
		for (int j = 0; j < numberOfObjects; j++) {
			printf("%f ", aijMatrix[i][j]);
			//printf("j has the value %f and is stored at %p\n", aijMatrix[i][j], (void *)&aijMatrix[i][j]);
			//printf("%f jjj ",aijMatrix[i][j]);
		}
		printf("\n");
	}
}

void printG(float* h_A, int numberOfPersons, int numberOfObjects) {
	for (int i = 0; i < numberOfPersons; i++) {
		for (int j = 0; j < numberOfObjects; j++) {
			printf("%f ", h_A[i * numberOfPersons + j]);
			//printf("j has the value %f and is stored at %p\n", aijMatrix[i][j], (void *)&aijMatrix[i][j]);
			//printf("%f jjj ",aijMatrix[i][j]);
		}
		printf("\n");
	}
}

void listCudaDevice() {
	int deviceCount;

	cudaGetDeviceCount(&deviceCount);

	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
		printf("There is no device supporting CUDA\n");
	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		if (dev == 0) {
			// This function call returns 9999 for both major & minor fields, if no CUDA capable devices are present
			if (deviceProp.major == 9999 && deviceProp.minor == 9999)
				printf("There is no device supporting CUDA.\n");
			else if (deviceCount == 1)
				printf("There is 1 device supporting CUDA\n");
			else
				printf("There are %d devices supporting CUDA\n", deviceCount);
		}
		printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);
		printf("  CUDA Capability Major revision number:         %d\n",
				deviceProp.major);
		printf("  CUDA Capability Minor revision number:         %d\n",
				deviceProp.minor);
		printf("  Total amount of global memory:                 %f bytes\n",
				deviceProp.totalGlobalMem);
#if CUDART_VERSION >= 2000
		printf("  Number of multiprocessors:                     %d\n", deviceProp.multiProcessorCount);
		printf("  Number of cores:                               %d\n", 8 * deviceProp.multiProcessorCount);
#endif
		printf("  Total amount of constant memory:               %u bytes\n",
				deviceProp.totalConstMem);
		printf("  Total amount of shared memory per block:       %u bytes\n",
				deviceProp.sharedMemPerBlock);
		printf("  Total number of registers available per block: %d\n",
				deviceProp.regsPerBlock);
		printf("  Warp size:                                     %d\n",
				deviceProp.warpSize);
		printf("  Maximum number of threads per block:           %d\n",
				deviceProp.maxThreadsPerBlock);
		printf(
				"  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
				deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
				deviceProp.maxThreadsDim[2]);
		printf(
				"  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
				deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
				deviceProp.maxGridSize[2]);
		printf("  Maximum memory pitch:                          %u bytes\n",
				deviceProp.memPitch);
		//		printf("  Texture alignment:                             %u bytes\n",
		//				deviceProp.textureAlignment);
		printf("  Clock rate:                                    %.2f GHz\n",
				deviceProp.clockRate * 1e-6f);
#if CUDART_VERSION >= 2000
		printf("  Concurrent copy and execution:                 %s\n", deviceProp.deviceOverlap ? "Yes" : "No");
#endif
#if CUDART_VERSION >= 2020
		printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
		printf("  Integrated:                                    %s\n", deviceProp.integrated ? "Yes" : "No");
		printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
		printf("  Compute mode:                                  %s\n", deviceProp.computeMode == cudaComputeModeDefault ?
				"Default (multiple host threads can use this device simultaneously)" :
				deviceProp.computeMode == cudaComputeModeExclusive ?
				"Exclusive (only one host thread at a time can use this device)" :
				deviceProp.computeMode == cudaComputeModeProhibited ?
				"Prohibited (no host thread can use this device)" :
				"Unknown");
#endif
	}

}

void checkCUDAError(const char *msg) {
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}
