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
#endif CUDPP

// Include C files
#include "AssignmentEngine.h"
#include "Generator.h"
// include kernels
#include "BestDiffKernelGlobalGPU.cu"
#include "BestDiffKernelSharedGPU.cu"
#include "SortKernel.cu"
#include "SwitchingKernel.cu"

//#include "InitAssignmentKernel.cu"

#define BLOCK_SIZE 16
#define DEFAULT_MULTI 8

typedef struct {
	float happiness;
	float time;
} TestResult;

// CUDA related
void checkCUDAError(const char *msg);
void listCudaDevice();

// CUDA Kernels
//__global__ void initialAssignment(float* diffs, int* bestChanges, AijMatrix A,
//		int* persons, int* objects, bool* bannedSwitches,
//		bool* clearedBannedSwitches, int* reset, int* blockSize);

__global__ void bestDiff(AijMatrix A, Difference* diffs, float* srtDiffs,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches);

__global__ void first(AijMatrix A, Difference* differences,
		Difference* rowDifferences, Difference* columnDifferences,
		float* srtDiffs, int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, Difference ret);

__global__ void placeDifferencesByIndex(Difference* diffs, int* index, int max);

//__global__ void
//findMax(Difference* diffs, int* bestChanges, AijMatrix A, int* persons,
//		int* objects, bool* bannedSwitches, bool* clearedBannedSwitches,
//		int* reset, int blockSize);

__global__ void findMaxShared(Difference* diffs, float* srtDiffs, AijMatrix A,
		int* persons, int* objects, bool* bannedSwitches,
		bool* clearedBannedSwitches, float* resDiffs);

__global__ void
bestDiffShared(Difference* diffs, AijMatrix A, int* persons, int* objects,
		bool* bannedSwitches, bool* clearedBannedSwitches, float* resDiffs);

// Host
TestResult runTest(int, int);
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
bool* bannedSwitches;
bool* clearedBannedSwitches;
Difference* differences;
//Difference* differences_temp;
Difference* columnDifferences;
Difference* rowDifferences;
Difference emptyDiff;
int numberOfPersons, numberOfObjects;
int* reset;
float* h_Diffs;
int* h_bestChanges;
float* h_srtRes;

// Variables on GPU
AijMatrix d_aijM;
int* d_persons;
int* d_objects;
bool* d_bannedSwitches;
bool* d_clearedBannedSwitches;
Difference* d_differences;
Difference* differences_temp;
Difference* d_columnDifferences;
Difference* d_rowDifferences;
Difference d_emptyDiff;
float* d_DiffResults;
float* d_srtDiffs;
int d_numberOfPersons, d_numberOfObjects;

//Sizes
unsigned int tAijMSizeB;
unsigned int tPersonsSizeB;
unsigned int tObjectSizeB;
unsigned int tbestChangesB;
unsigned int tDiffsSizeB;
unsigned int tSrtDiffs;
unsigned int tBannedSwitches;
unsigned int tClearedBannedSwitches;
unsigned int indexSize;
//AijMatrix d_aijM;
//int* d_pers;
//int* d_objs;
//Difference d_emptyDiff;
//float* d_Diffs;
//int* d_bestChanges;
//int* d_reset;
//bool* d_bannedSwitches;
//bool* d_clearedBannedSwitches;
//int* d_blockSize;
int* h_index;
int* d_index;
//float* d_DiffResults;

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

// Timers for benchmarking
float tEvaluateDiff = 0.0f;
float tSorting = 0.0f;
float tSwitching = 0.0f;
int iterations = 0;

int minMult = 0;
int blockSize = BLOCK_SIZE;

int maxMult = 10;

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

	//checkCUDAError("cudaGetDeviceProperties");

	if (!deviceProp.canMapHostMemory)
	{
		fprintf(stderr, "Device %d cannot map host memory!\n", 0);
		exit( EXIT_FAILURE);
	}

	//	printf("%f%% free, %f%% used\n", 100.0 * free / (double) total, 100.0
	//			* (total - free) / (double) total);
}

// Main
int main(int argc, char** argv) {
	for (int i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-l") == 0) /* Process optional arguments. */
		{
			listCudaDevice();
			return 0;
		}
		if (strcmp(argv[i], "-c") == 0) /* Process optional arguments. */
		{
			runCpu = true;
			runGpu = false;

		}
		if (strcmp(argv[i], "-ni") == 0) /* Process optional arguments. */
		{
			niOut = true;

		}
		if (strcmp(argv[i], "-t") == 0) /* Process optional arguments. */
		{
			pTimer = true;
		}
		if (strcmp(argv[i], "-ag") == 0) /* Process optional arguments. */
		{
			assignmentGpu = true;
			assignmentCpu = false;
		}
		if (strcmp(argv[i], "-cg") == 0) /* Process optional arguments. */
		{
			runCpu = true;
		}

		if (strcmp(argv[i], "-b") == 0) /* Process optional arguments. */
		{
			blockSize = atoi(argv[i + 1]);
		}

		if (strcmp(argv[i], "-ao") == 0) /* Process optional arguments. */
		{
			assignmentOnly = true;
		}

		if (strcmp(argv[i], "-m") == 0) /* Process optional arguments. */
		{
			multi = atoi(argv[i + 1]);
		}
		if (strcmp(argv[i], "-p") == 0) /* Process optional arguments. */
		{
			pOut = true;
		}
		if (strcmp(argv[i], "-ri") == 0) /* Process optional arguments. */
		{
			pResInit = true;
		}
		if (strcmp(argv[i], "-ra") == 0) /* Process optional arguments. */
		{
			pResAss = true;
		}
		if (strcmp(argv[i], "-d") == 0) /* Process optional arguments. */
		{
			pDbg = true;
		}
		if (strcmp(argv[i], "-gen") == 0) /* Process optional arguments. */
		{
			useGenerator = true;
		}
		if (strcmp(argv[i], "-sg") == 0) /* Process optional arguments. */
		{
			sGPU = true;
		}
		if (strcmp(argv[i], "-so") == 0) /* Process optional arguments. */
		{
			sortP = true;
		}
		if (strcmp(argv[i], "-sdk") == 0) /* Process optional arguments. */
		{
			sdk = true;
		}
		if (strcmp(argv[i], "-mT") == 0) /* Process optional arguments. */
		{
			mTests = true;
			minMult = atoi(argv[i + 1]);
			maxMult = atoi(argv[i + 2]);
		}

	}

	//	// use command-line specified CUDA device, otherwise use device with highest Gflops/s
	//	if (cutCheckCmdLineFlag(argc, (const char**) argv, "device"))
	//		cutilDeviceInit(argc, argv);
	//	else
	int GPU_N;
	cutilSafeCall(cudaGetDeviceCount(&GPU_N));

	if (!niOut)
	{
		printf("CUDA-capable device count: %i\n", GPU_N);
	}
	for (int i = 0; i < GPU_N; i++)
	{
		CUdevice device;
		cuDeviceGet(&device, i);
		CUcontext ctx;
		cuCtxCreate(&ctx, 0, device);
		CUresult res = cuMemGetInfo(&freeMemDevice, &totalMemDevice);
		if (!niOut)
		{
			printStats(i, freeMemDevice, totalMemDevice);
		}
	}

	/*
	 * Check memory available
	 */

	if (!mTests)
	{
		minMult = multi;
		maxMult = multi + 1;
	}
	for (int var = minMult; var < maxMult; ++var)
	{
		numberOfPersons = blockSize * var;
		numberOfObjects = blockSize * var;
		TestResult r = runTest(numberOfPersons, numberOfObjects);
		if (!assignmentOnly)
			printf("%d, %f, %f, %f, %f, %f\n", numberOfPersons, r.happiness,
					r.time, (tEvaluateDiff / (float) iterations), (tSorting
							/ (float) iterations), (tSwitching
							/ (float) iterations));
	}
	return 0;
}

TestResult runTest(int numberOfPersons, int numberOfObjects) {
	if (useGenerator)
	{
		if (!niOut)
		{
			printf("Using Euclidean Generator\n");
		}
		aijMatrix = genMatrix(numberOfPersons, numberOfObjects);
	} else
	{
		if (!niOut)
		{
			printf("Using Random Generator\n");
		}
		// For testing purpose only
		int C = 3000;
		aijMatrix = (float **) malloc(numberOfPersons * sizeof(float *));

		float *aijPtr = (float *) malloc(numberOfPersons * numberOfObjects
				* sizeof(float));
		for (int i = 0; i < numberOfPersons; i++)
		{
			aijMatrix[i] = aijPtr + (i * numberOfObjects);
		}

		for (int i = 0; i < numberOfPersons; i++)
		{
			for (int j = 0; j < numberOfObjects; j++)
			{
				aijMatrix[i][j] = random() % C;
			}
		}

	}

	//	int *initialAssignment;
	//	initialAssignment = (int *) malloc(numberOfPersons * sizeof(int));
	//	int len = 0;
	//	if (len > 0)
	//	{
	//		for (int i = 0; i < numberOfPersons; i++)
	//		{
	//			initialAssignment[i] = i;
	//		}
	//	}

	//RUN
	TestResult r = runHeuristic(aijMatrix, numberOfPersons, numberOfObjects);

	return r;
}

/**
 * Initialize structure on host memory
 */
void hostInit(float** aijMatrix, int numberOfPersons, int numberOfObjects) {

	emptyDiff.index = -1;
	emptyDiff.myAssigned = -1;
	emptyDiff.bestChangeAssigned = -1;

	bannedSwitches = (bool *) malloc(numberOfPersons * numberOfPersons
			* sizeof(bool));

	int i, j;
	clearedBannedSwitches = (bool *) malloc(numberOfPersons * sizeof(bool));
	persons = (int *) malloc(numberOfPersons * sizeof(int));
	differences = (Difference *) malloc((numberOfPersons + numberOfObjects)
			* sizeof(Difference));
	rowDifferences
			= (Difference *) malloc(numberOfPersons * sizeof(Difference));
	for (i = 0; i < numberOfPersons; i++)
	{
		clearedBannedSwitches[i] = false;
		persons[i] = -1;
		for (j = 0; j < numberOfPersons; j++)
			bannedSwitches[i * numberOfObjects + j] = false;
		rowDifferences[i] = emptyDiff;
		differences[i] = emptyDiff;
	}

	columnDifferences = (Difference *) malloc(numberOfObjects
			* sizeof(Difference));
	objects = (int *) malloc(numberOfObjects * sizeof(int));
	for (i = 0; i < numberOfObjects; i++)
	{
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
	unsigned int tRowDiffsSizeB = sizeof(Difference) * (numberOfPersons);
	unsigned int tColDiffsSizeB = sizeof(Difference) * (numberOfObjects);
	//	int resetB = sizeof(bool);
	tClearedBannedSwitches = sizeof(bool) * numberOfPersons;

	unsigned int tAijMSize = numberOfPersons * numberOfObjects;
	tAijMSizeB = sizeof(float) * tAijMSize;
	tBannedSwitches = sizeof(bool) * tAijMSize;

	int totalBonGpu = tPersonsSizeB + tObjectSizeB + tAijMSizeB + tDiffsSizeB
			+ tBannedSwitches + tClearedBannedSwitches;

	if (!niOut)
	{
		printf("Memory used on GPU: %d Bytes\n", totalBonGpu);
	}

	if (totalBonGpu > freeMemDevice)
	{
		printf("Warning: not enough memory available on GPU: %d Bytes\n",
				freeMemDevice);
	}

	float* h_aijM = (float*) malloc(tAijMSizeB);

	for (int i = 0; i < numberOfPersons; i++)
	{
		for (int j = 0; j < numberOfObjects; j++)
		{
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
	cutilSafeCall(cudaMalloc((void**) &d_persons, tPersonsSizeB));
	//object
	cutilSafeCall(cudaMalloc((void**) &d_objects, tObjectSizeB));
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
	cutilSafeCall(cudaMemcpy(d_persons, persons, tPersonsSizeB,
			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_objects, objects, tObjectSizeB,
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

	cutilSafeCall(cudaMalloc((void**) &d_rowDifferences, tRowDiffsSizeB));
	cutilSafeCall(cudaMalloc((void**) &d_columnDifferences, tColDiffsSizeB));

	cutilSafeCall(cudaMemcpy(d_differences, differences, tDiffsSizeB,
			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_rowDifferences, rowDifferences, tRowDiffsSizeB,
			cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_columnDifferences, columnDifferences,
			tColDiffsSizeB, cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_bestChanges, h_bestChanges, tbestChangesB,
	//			cudaMemcpyHostToDevice));

	if (sdk)
	{
		//ALLOCATE ROOM FOR THE RESULTING DIFFERENCES
		cutilSafeCall(cudaMalloc((void**) &d_DiffResults, tAijMSizeB));
	}

	cuMemGetInfo(&freeMemDevice, &totalMemDevice);

	if (!niOut)
		printf("Memory after allocation: %f%% free, %f%% used\n", 100.0
				* freeMemDevice / (double) totalMemDevice, 100.0
				* (totalMemDevice - freeMemDevice) / (double) totalMemDevice);
}

void gpuTerninate() {
	cudaFree(d_aijM.els);
	cudaFree(d_persons);
	cudaFree(d_objects);
	cudaFree(d_differences);
	cudaFree(d_bannedSwitches);
	cudaFree(d_clearedBannedSwitches);
}

TestResult runHeuristic(float** aijMatrix, int numberOfPersons,
		int numberOfObjects) {

	//Init
	hostInit(aijMatrix, numberOfPersons, numberOfObjects);

	//	if (len > 0)
	//		smartInitialAssignmentWithInitial( initialAssignment);
	//	else
	//		smartInitialAssignment();
	//	enhanceBySwitching();
	if (!niOut)
	{
		printf("Entities %d\n", numberOfPersons);
		printf("Block Size %d\n", blockSize);
	}
	unsigned int timer = 0;
	cutilCheckError(cutCreateTimer(&timer));
	if (assignmentGpu)
	{
		gpuInit(aijMatrix, numberOfPersons, numberOfObjects);
		if (!niOut)
		{
			printf("-----GPU Assignment------\n");
		}
		smartInitialAssignmentGPU();
	}

	if (assignmentCpu)
	{
		cutilCheckError(cutCreateTimer(&timer));
		cutilCheckError(cutStartTimer(timer));
		if (!niOut)
		{

			printf("-----CPU Assignment------\n");
		}
		smartInitialAssignment();

		cutilCheckError(cutStopTimer(timer));
		if (!niOut)
		{
			printf("Done Assignment\n");
			printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
		}
	}

	if (pResInit)
	{
		for (int i = 0; i < numberOfPersons; i++)
		{
			printf("aij ");
			printf("%d ", i);
			printf("%d ", persons[i]);
			printf("%f \n", aijMatrix[i][persons[i]]);
		}

	}

	TestResult r;
	if (!assignmentOnly)
	{

		if (!niOut)
		{
			printf("Enhance By Switching ");
		}
		if (runGpu)
		{
			if (!niOut)
			{
				printf("on GPU\n");
			}
			if (!assignmentGpu)
			{
				gpuInit(aijMatrix, numberOfPersons, numberOfObjects);
			}
		} else
		{
			if (!niOut)
			{
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
		if (!niOut)
		{
			printf("Done Enhancing\n");
			printf("Processing time: %f (ms)\n", tValue);
		}
		//		cutilCheckError(cutStopTimer(timer));
		//		printf("Processing time: %f (ms)\n", cutGetTimerValue(timer));
		if (pResAss)
		{
			for (int i = 0; i < numberOfPersons; i++)
			{
				printf("aij ");
				printf("%d ", i);
				printf("%d ", persons[i]);
				printf("%f \n", aijMatrix[i][persons[i]]);
			}

		}
		r.time = tValue;
		r.happiness = happiness;
	}

	if (runGpu || assignmentGpu)
	{
		if (!niOut && !assignmentOnly)
		{
			printf("Cleaning GPU state\n");
		}
		gpuTerninate();
	}

	return r;

}

void gpuInit2() {

	//	// Pass result form teh smart initial assignment
	//	cutilSafeCall(cudaMemcpy(d_persons, persons, tPersonsSizeB,
	//			cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_objects, objects, tObjectSizeB,
	//			cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_bannedSwitches, bannedSwitches, tBannedSwitches,
	//			cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_clearedBannedSwitches, clearedBannedSwitches,
	//			tClearedBannedSwitches, cudaMemcpyHostToDevice));

	//	cutilSafeCall(cudaMemcpy(d_bannedSwitches, bannedSwitches, tBannedSwitches,
	//			cudaMemcpyHostToDevice));
	//	cutilSafeCall(cudaMemcpy(d_clearedBannedSwitches, clearedBannedSwitches,
	//			tClearedBannedSwitches, cudaMemcpyHostToDevice));

	int size = numberOfPersons + numberOfObjects;
	h_srtRes = (float*) malloc(sizeof(float) * size);

	for (int var = 0; var < size; ++var)
	{
		h_srtRes[var] = 0.0f;
	}

	cutilSafeCall(cudaMemcpy(d_srtDiffs, h_srtRes, sizeof(float) * size,
			cudaMemcpyHostToDevice));

#ifdef CUDPP

	if(sGPU)
	{
		int size = numberOfPersons + numberOfObjects;
		CUDPPConfiguration config;
		config.algorithm = CUDPP_SORT_RADIX;
		config.datatype = CUDPP_FLOAT;
		config.options = CUDPP_OPTION_KEY_VALUE_PAIRS;

		CUDPPResult result = cudppPlan(&scanplan, config, size, 1, 0);

		if (CUDPP_SUCCESS != result)
		{
			string eMsg = "Error creating CUDPPPlan";
			fail(eMsg);
		}
		indexSize = (size) * sizeof(int);
		h_index = (int*) malloc(indexSize);
		cutilSafeCall(cudaMalloc((void**) &d_index, indexSize));

	}

	//	free(h_Diffs);

	//	tDiffsSizeB = (numberOfPersons + numberOfObjects) * sizeof(float);
	//	h_Diffs = (float*) malloc(tDiffsSizeB);

	//	for (int var = 0; var < numberOfPersons + numberOfObjects; ++var)
	//	{
	//		h_Diffs[var] = 0.0f;
	//	}

	//	cutilSafeCall(cudaMalloc((void**) &d_Diffs, tDiffsSizeB));
	//
	//	cutilSafeCall(cudaMemcpy(d_Diffs, h_Diffs, tDiffsSizeB,
	//			cudaMemcpyHostToDevice));

	//	differences_temp = (Difference *) malloc(
	//			(numberOfPersons + numberOfObjects) * sizeof(Difference));
#endif
}

void evaluateDifferences() {

	if (runGpu)
	{
		if (pDbg)
		{
			printf("Eval Diff Phase Start\n");
		}

		dim3 threads;
		dim3 grid;

		if (!sdk)
		{
			threads.x = blockSize;
			threads.y = 1;
			int s = numberOfPersons / blockSize + (numberOfPersons % blockSize
					== 0 ? 0 : 1);
			grid.x = s;
			grid.y = 1;
		} else
		{
			threads.x = blockSize;
			threads.y = blockSize;
			grid.x = numberOfObjects / threads.x;
			grid.y = numberOfPersons / threads.y;
		}

		//		cutilSafeCall(cudaMemcpy(d_pers, persons, tPersonsSizeB,
		//				cudaMemcpyHostToDevice));
		//		cutilSafeCall(cudaMemcpy(d_objs, objects, tObjectSizeB,
		//				cudaMemcpyHostToDevice));
		//		cutilSafeCall(cudaMemcpy(d_bannedSwitches, bannedSwitches,
		//				tBannedSwitches, cudaMemcpyHostToDevice));
		//		cutilSafeCall(cudaMemcpy(d_clearedBannedSwitches,
		//				clearedBannedSwitches, tClearedBannedSwitches,
		//				cudaMemcpyHostToDevice));

		if (pDbg)
		{
			printf("Pre-MemCpy Over\n");
		}
		//		reset[0] = 1;
		//		cutilSafeCall(cudaMemcpy(d_reset, reset, sizeof(int),
		//				cudaMemcpyHostToDevice));
		unsigned int timerEDSub = 0;
		cutilCheckError(cutCreateTimer(&timerEDSub));
		unsigned int timerEDSubb = 0;
		cutilCheckError(cutCreateTimer(&timerEDSubb));
		if (sdk)
		{
			cutilCheckError(cutStartTimer(timerEDSub));
			//			if (!niOut)
			//				printf("GRID %d %d TH %d %d \n", grid.y, grid.x, threads.x,
			//						threads.y);
			bestDiffShared <<<grid,threads>>>(d_differences, d_aijM, d_persons, d_objects,d_bannedSwitches,d_clearedBannedSwitches,d_DiffResults);
			CUT_CHECK_ERROR("bestDiff");
			cutilCheckError(cutStopTimer(timerEDSub));
			if (pTimer)
				printf("EDSub time: %f (ms)\n", cutGetTimerValue(timerEDSub));
			cutilCheckError(cutStartTimer(timerEDSubb));
			threads.x = blockSize;
			threads.y = 1;
			int s = numberOfPersons / blockSize + (numberOfPersons % blockSize
					== 0 ? 0 : 1);
			grid.x = s;
			grid.y = 1;
			findMaxShared <<<grid,threads>>>(d_differences,d_srtDiffs, d_aijM, d_persons, d_objects,d_bannedSwitches,d_clearedBannedSwitches,d_DiffResults);
			cutilCheckError(cutStopTimer(timerEDSubb));
			if (pTimer)
				printf("EDSubb time: %f (ms)\n", cutGetTimerValue(timerEDSubb));
		} else
		{
			if (!niOut)
				printf("GRID %d %d TH %d %d \n", grid.y, grid.x, threads.x,
						threads.y);
			cutilCheckError(cutStartTimer(timerEDSub));
			bestDiff<<<grid,threads>>>( d_aijM, d_differences,d_srtDiffs,d_persons, d_objects,d_bannedSwitches,d_clearedBannedSwitches);
			cutilCheckError(cutStopTimer(timerEDSub));
			if (pTimer)
				printf("EDSub time: %f (ms)\n", cutGetTimerValue(timerEDSub));
			cutilCheckError(cutStartTimer(timerEDSubb));
			CUT_CHECK_ERROR("bestDiff");
			//			findMax<<<grid,threads>>>(d_Diffs, d_bestChanges,d_aijM, d_pers, d_objs,d_bannedSwitches,d_clearedBannedSwitches,d_reset,blockSize);
			//			cutilCheckError(cutStopTimer(timerEDSubb));
			//			if (pTimer)
			//				printf("EDSubb time: %f (ms)\n", cutGetTimerValue(timerEDSubb));
			//			CUT_CHECK_ERROR("findMax");
		}
		cudaThreadSynchronize();
		//		cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
		//				cudaMemcpyDeviceToHost));

		//		cutilSafeCall(cudaMemcpy(h_bestChanges, d_bestChanges, tbestChangesB,
		//				cudaMemcpyDeviceToHost));
		//		cutilSafeCall(cudaMemcpy(persons, d_persons, tPersonsSizeB,
		//				cudaMemcpyDeviceToHost));

		if (pDbg)
		{
			printf("Post-MemCpy Over\n");
		}

		//		for (int i = 0; i < numberOfPersons; i++)
		//		{
		//			//int myCol = persons[i];
		//			float maxDiff = h_Diffs[i];
		//			//printf("bc: %d\n",h_bestChanges[i]);
		//			int bestChangeCol = h_bestChanges[i];
		//			Difference curDiff;
		//			/*if (maxDiff < 0)
		//			 maxDiff = -maxDiff;
		//			 if (maxDiff > 0.1 || myCol == -1)
		//			 {
		//			 if (bestChangeCol == -1)
		//			 {
		//			 if (clearedBannedSwitches[i])
		//			 {
		//			 persons[i] = -1;
		//			 curDiff = emptyDiff;
		//			 } else
		//			 {
		//			 clearedBannedSwitches[i] = true;
		//			 int x;
		//			 for (x = 0; x < numberOfPersons; x++)
		//			 bannedSwitches[i * numberOfObjects + x] = false;
		//
		//			 curDiff = getRowBestDifference(i);
		//			 }
		//			 } else
		//			 {
		//			 if (myCol == -1)
		//			 maxDiff = maxDiff * 1000;
		//
		//
		//			 curDiff.value = maxDiff;
		//			 }
		//			 } else
		//			 {
		//			 curDiff = emptyDiff;
		//			 }*/
		//
		//			if (maxDiff != negInf)
		//			{
		//				curDiff.index = i;
		//				curDiff.bestChange = bestChangeCol;
		//				curDiff.type = 0;
		//				curDiff.myAssigned = persons[i];
		//				curDiff.bestChangeAssigned = objects[bestChangeCol];
		//				curDiff.value = maxDiff;
		//				differences[i] = curDiff;
		//				rowDifferences[i] = curDiff;
		//				//printf("ass diff[%d]=%f bc %d bca:%d ma:%d\n",i,maxDiff, bestChangeCol,curDiff.bestChangeAssigned,curDiff.myAssigned);
		//			}
		//		}
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

	for (int var = 0; var < numberOfPersons + numberOfObjects; ++var)
	{
		h_index[var] = var;
		//			if (differences[var].index == -1)
		//			h_Diffs[var] = negInf;
		//			else
		//			h_Diffs[var] = differences[var].value;
		//		printf("i[%d]=%d\n",var,h_index[var]);
	}

	if (pDbg)
	{
		printf("Pre-sort\n");
	}
	//
	//	cutilSafeCall(cudaMemcpy(d_Diffs, h_Diffs, tDiffsSizeB,
	//					cudaMemcpyHostToDevice));
	cutilSafeCall(cudaMemcpy(d_index, h_index, indexSize,
					cudaMemcpyHostToDevice));

	//	cutilSafeCall(cudaMemcpy(h_srtRes, d_srtDiffs, sizeof(float)*size,
	//					cudaMemcpyDeviceToHost));
	//
	//	for (int var = 0; var < size; ++var)
	//	{
	//		printf("i[%d]=%d %f\n",var,h_index[var],h_srtRes[var]);
	//	}
	// Sorting now
	CUDPPResult res = cudppSort(scanplan, d_srtDiffs,d_index, 32,size);

	if (CUDPP_SUCCESS != res)
	{
		string eMsg = "Error sorting on GPU\n";
		fail(eMsg);
	}
	CUT_CHECK_ERROR("cudppRadixSort");

	if (pDbg)
	{
		printf("Post-sort\n");
	}

	//	cutilSafeCall(cudaMemcpy(h_index, d_index, indexSize,
	//					cudaMemcpyDeviceToHost));

	//	cutilSafeCall(cudaMemcpy(h_index, d_index, indexSize,
	//					cudaMemcpyDeviceToHost));
	//	cutilSafeCall(cudaMemcpy(h_srtRes, d_srtDiffs, sizeof(float)*size,
	//					cudaMemcpyDeviceToHost));
	//
	//	for (int var = 0; var < size; ++var)
	//	{
	//		printf("i[%d]=%d %f\n",var,h_index[var],h_srtRes[var]);
	//	}

	dim3 threads;
	dim3 grid;

	threads.x = blockSize;
	threads.y = 1;
	int s = (numberOfPersons+numberOfObjects) / blockSize + ((numberOfPersons+numberOfObjects) % blockSize
			== 0 ? 0 : 1);
	grid.x = s;
	grid.y = 1;

	placeDifferencesByIndex<<<grid,threads>>>(d_differences,d_index,size);

	//	for (int curEntry = 0; curEntry < (size/2); curEntry++)
	//	{
	//		int fI=curEntry;
	//		int sI=(size - 1 - curEntry);
	//		Difference temp=differences[fI];
	//		printf("cur diff[%d]=%f\n",fI,temp.value);
	//		printf("other diff[%d]=%f\n",sI,differences[h_index[sI]].value);
	//		differences[curEntry]=differences[h_index[sI]];
	//		differences[sI]=temp;
	//		//		differences_temp[curEntry] = differences[h_index[size - 1 - curEntry]];
	//		Difference curDiff=differences[curEntry];
	//		printf("ass diff[%d]=%f bc %d bca:%d ma:%d\n\n",curEntry,
	//				curDiff.value, curDiff.bestChange,
	//				curDiff.bestChangeAssigned, curDiff.myAssigned);
	//	}
	//
	//	for (int varT = 0; varT < size; varT++)
	//	{
	//
	//		if (varT == size - 1)
	//		{
	//			break;
	//		}
	//		float cv = differences[varT].value;
	//
	//		if (cv != 0.0f)
	//		{
	//			int count = 0;
	//			for (int var1 = varT + 1; var1 < size; ++var1)
	//			{
	//				if (differences[var1].value == cv)
	//				{
	//					count++;
	//				} else
	//				break;
	//			}
	//			if (count > 0)
	//			{
	//				Difference ds[count+1];
	//				int inc = varT + count;
	//				for (int N = 0; N < count + 1; ++N)
	//				{
	//					ds[N] = differences[varT + N];
	//				}
	//				qsort(ds, count + 1, sizeof(Difference), compare_differences);
	//				for (int N = 0; N < count + 1; ++N)
	//				{
	//					differences[varT + N] = ds[N];
	//				}
	//				varT = varT + count;
	//			}
	//		}
	//
	//	}

	if (pDbg)
	{
		printf("Post-red\n");
	}
#endif
}

void fail(string& s) {
	cout << s;
	exit(-1);
}

void smartInitialAssignmentGPU() {
	//	unsigned int timerAGPU = 0;
	//	cutilCheckError(cutCreateTimer(&timerAGPU));
	//	cutilCheckError(cutStartTimer(timerAGPU));
	//
	//	int s = numberOfPersons / blockSize + (numberOfPersons % blockSize == 0 ? 0
	//			: 1);
	//	dim3 grid(s);
	//
	//	reset[0] = 1;
	//	cudaMemcpy(d_Diffs, h_Diffs, tDiffsSizeB, cudaMemcpyHostToDevice);
	//	cudaMemcpy(d_bestChanges, h_bestChanges, tbestChangesB,
	//			cudaMemcpyHostToDevice);
	//
	//	unsigned int timer = 0;
	//	cutilCheckError(cutCreateTimer(&timer));
	//	cutilCheckError(cutResetTimer(timer));
	//
	//	reset[0] = 1;
	//	while (reset[0] == 1)
	//	{
	//		cutilCheckError(cutResetTimer(timer));
	//		cutilCheckError(cutStartTimer(timer));
	//		reset[0] = 0;
	//		cutilSafeCall(cudaMemcpy(d_reset, reset, sizeof(bool),
	//				cudaMemcpyHostToDevice));
	//		bestDiff<<<grid,blockSize>>>(d_Diffs,d_bestChanges, d_aijM, d_pers, d_objs,d_bannedSwitches,d_clearedBannedSwitches,d_reset,blockSize);
	//		cudaThreadSynchronize();
	//		cutilCheckError(cutStopTimer(timer));
	//		if (pTimer)
	//			printf("BD time: %f (ms)\n", cutGetTimerValue(timer));
	//
	//		cutilCheckError(cutResetTimer(timer));
	//		cutilCheckError(cutStartTimer(timer));
	//		reset[0] = 0;
	//		cutilSafeCall(cudaMemcpy(d_reset, reset, sizeof(bool),
	//				cudaMemcpyHostToDevice));
	//		cutilCheckError(cutStopTimer(timer));
	//		//		if (pTimer)
	//		//			printf("CR time: %f (ms)\n", cutGetTimerValue(timer));
	//		//Kernel here
	//		//	dim3 dimBlock(1,numberOfPersons);
	//		//	dim3 dimGrid(1, 1);
	//		cutilCheckError(cutResetTimer(timer));
	//		cutilCheckError(cutStartTimer(timer));
	//		initialAssignment<<<grid,1>>>(d_Diffs,d_bestChanges,d_aijM, d_pers, d_objs,d_bannedSwitches,d_clearedBannedSwitches,d_reset,d_blockSize);
	//		cudaThreadSynchronize();
	//		cutilCheckError(cutStopTimer(timer));
	//		if (pTimer)
	//			printf("IA time: %f (ms)\n", cutGetTimerValue(timer));
	//
	//		cutilCheckError(cutResetTimer(timer));
	//		cutilCheckError(cutStartTimer(timer));
	//		cudaMemcpy(reset, d_reset, sizeof(int), cudaMemcpyDeviceToHost);
	//		cutilCheckError(cutStopTimer(timer));
	//		if (pTimer)
	//			printf("CR time: %f (ms)\n", cutGetTimerValue(timer));
	//
	//	}
	//
	//	unsigned int timerMcpy = 0;
	//	cutilCheckError(cutCreateTimer(&timerMcpy));
	//	cutilCheckError(cutStartTimer(timerMcpy));
	//
	//	cudaMemcpy(h_Diffs, d_Diffs, tDiffsSizeB, cudaMemcpyDeviceToHost);
	//	cudaMemcpy(h_bestChanges, d_bestChanges, tbestChangesB,
	//			cudaMemcpyDeviceToHost);
	//
	//	cutilSafeCall(cudaMemcpy(persons, d_pers, tPersonsSizeB,
	//			cudaMemcpyDeviceToHost));
	//
	//	cutilSafeCall(cudaMemcpy(objects, d_objs, tObjectSizeB,
	//			cudaMemcpyDeviceToHost));
	//
	//	cutilSafeCall(cudaMemcpy(bannedSwitches, d_bannedSwitches, tBannedSwitches,
	//			cudaMemcpyDeviceToHost));
	//	cutilSafeCall(cudaMemcpy(clearedBannedSwitches, d_clearedBannedSwitches,
	//			tClearedBannedSwitches, cudaMemcpyDeviceToHost));
	//
	//	cutilCheckError(cutStopTimer(timerMcpy));
	//	if (pTimer)
	//		printf("MCR time: %f (ms)\n", cutGetTimerValue(timerMcpy));
	//
	//	cutilCheckError(cutStopTimer(timerAGPU));
	//	printf("Assignment time: %f (ms)\n", cutGetTimerValue(timerAGPU));
	//
	//	if (pOut)
	//	{
	//		for (int k = 0; k < numberOfPersons; k++)
	//		{
	//			printf("P%d O%d ->%f \n", k, persons[k], h_Diffs[k]);
	//		}
	//	}
}

void enhanceBySwitching() {
	// int counter = 1;
	//  time_t start = time(NULL);

	if (runGpu)
	{
		gpuInit2();
	}
	if (!niOut)
	{
		if (sGPU)
		{
			printf("Sorting on GPU\n");
		} else
		{
			printf("Sorting on CPU \n");
		}
	}

	//	int size = numberOfPersons + numberOfObjects;


	float oldTotalHappiness = calculateTotalHappiness();

	//	for (int var = 0; var < numberOfPersons; ++var)
	//	{
	//		printf("%d ->%d\n", var, persons[var]);
	//	}

	unsigned int timerED = 0;
	cutilCheckError(cutCreateTimer(&timerED));
	unsigned int timerSort = 0;
	cutilCheckError(cutCreateTimer(&timerSort));
	unsigned int timerProc = 0;
	cutilCheckError(cutCreateTimer(&timerProc));

	while (1)
	{
		cutilCheckError(cutResetTimer(timerED));
		cutilCheckError(cutResetTimer(timerSort));
		cutilCheckError(cutResetTimer(timerProc));

		cutilCheckError(cutStartTimer(timerED));
		//	int count = 0;
		//	while (1)
		//	{
		//		//		if (pDbg)
		//		//		{
		//		if (!niOut)
		//		{
		//			printf("\nIteration %d\n", count++);
		//		}
		//		iterations++;
		//		//		}
		//		oldTotalHappiness = calculateTotalHappiness();
		//
		//		cutilCheckError(cutStartTimer(timerED));
		//--------
		evaluateDifferences();
		cutilCheckError(cutStopTimer(timerED));
		//		if (pTimer)
		printf("ED time: %f (ms)\n", cutGetTimerValue(timerED));
		//--------
		//		cutilCheckError(cutStopTimer(timerED));
		//		if (pTimer)
		//			printf("ED time: %f (ms)\n", cutGetTimerValue(timerED));
		//		tEvaluateDiff += cutGetTimerValue(timerED);
		//		cutilCheckError(cutResetTimer(timerED));
		//
		//		int switchedRows[2];
		//		int switchedColumns[2];
		//
		//		if (sortP)
		//		{
		//			printf("Before sorting\n");
		//			int count2 = 0;
		//			for (int var = 0; var < numberOfPersons + numberOfObjects; ++var)
		//			{
		//				count2++;
		//				printf("\t %d i%d->%f", var, differences[var].index,
		//						differences[var].value);
		//				if (count2 == 8)
		//				{
		//					printf("\n");
		//					count2 = 0;
		//				}
		//			}
		//		}
		//
		cutilCheckError(cutStartTimer(timerSort));
		if (sGPU)
		{
			sortDifferencesGPU();
			//		cutilCheckError(cutStopTimer(timerED));
			//		if (pTimer)
			//			printf("ED time: %f (ms)\n", cutGetTimerValue(timerED));
		} else
		{
			cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
					cudaMemcpyDeviceToHost));
			qsort(differences, numberOfPersons + numberOfPersons,
					sizeof(Difference), compare_differences);
			cutilSafeCall(cudaMemcpy(d_differences, differences, tDiffsSizeB,
					cudaMemcpyHostToDevice));
		}
		cutilCheckError(cutStopTimer(timerSort));
		//		if (pTimer)
		printf("Srt time: %f (ms)\n", cutGetTimerValue(timerSort));
		//
		//if (sortP)
		//{
		//	if (sGPU)
		//		cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
		//				cudaMemcpyDeviceToHost));

		if (!niOut)
			printf("After sorting\n");
		//	int count2 = 0;
		//	for (int var = 0; var < numberOfPersons + numberOfObjects; ++var)
		//	{
		//		count2++;
		//		printf("\t %d i%d->%f", var, differences[var].index,
		//				differences[var].value);
		//		if (count2 == 8)
		//		{
		//			printf("\n");
		//			count2 = 0;
		//		}
		//	}
		//	}

		//	cutilCheckError(cutStopTimer(timerSort));
		//	if (pTimer)
		//		printf("Srt time: %f (ms)\n", cutGetTimerValue(timerSort));

		//	tSorting += cutGetTimerValue(timerSort);
		//	cutilCheckError(cutResetTimer(timerSort));
		//	int rdC = 0;
		//	cutilCheckError(cutStartTimer(timerProc));

		cutilCheckError(cutStartTimer(timerProc));

		dim3 threads;
		dim3 grid;

		threads.x = 1;
		threads.y = 1;
		int s = numberOfPersons / blockSize
				+ (numberOfPersons % blockSize == 0 ? 0 : 1);
		grid.x = 1;
		grid.y = 1;
		Difference ret;
		cutilSafeCall(cudaMalloc((void**) &ret, sizeof(Difference)));

		first<<<grid,threads>>>( d_aijM, d_differences,d_rowDifferences,d_columnDifferences, d_srtDiffs,d_persons, d_objects,d_bannedSwitches,d_clearedBannedSwitches,ret);

		//		cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
		//				cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaMemcpy(persons, d_persons, tPersonsSizeB,
				cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(objects, d_objects, tObjectSizeB,
				cudaMemcpyDeviceToHost));

		cutilCheckError(cutStopTimer(timerProc));

		printf("Proc time: %f (ms)\n", cutGetTimerValue(timerProc));

		float newTotalHappiness = calculateTotalHappiness();
		if (newTotalHappiness == oldTotalHappiness)
		{
			break;
		} else
			oldTotalHappiness = newTotalHappiness;

	}

	printf("Copying back result\n");

	//
	//	if (sortP)
	//	{
	//		cutilSafeCall(cudaMemcpy(differences, d_differences, tDiffsSizeB,
	//				cudaMemcpyDeviceToHost));
	//		printf("After enhancing\n");
	//		int count2 = 0;
	//		for (int var = 0; var < numberOfPersons + numberOfObjects; ++var)
	//		{
	//			count2++;
	//			printf("\t %d i%d->%f", var, differences[var].index,
	//					differences[var].value);
	//			if (count2 == 8)
	//			{
	//				printf("\n");
	//				count2 = 0;
	//			}
	//		}
	//	}

	//	if (pDbg)
	//	{
	//	for (int var = 0; var < numberOfPersons; ++var)
	//	{
	//		printf("%d ->%d\n", var, persons[var]);
	//	}
	//	}
	//	}
}

void smartInitialAssignment() {
	int row1, curRow, col2, i;

	for (i = 0; i < numberOfPersons; i++)
	{

		curRow = i;
		while (curRow != -1)
		{
			Difference myDiff = getRowBestDifference(curRow);
			if (myDiff.index != -1)
			{
				row1 = myDiff.index; // index of row of the difference
				col2 = myDiff.bestChange; // index of column of the best
				// cell in the row of difference
				curRow = objects[col2]; // index of row of the chosen in the
				// column of the best cell in the
				// difference row
				persons[row1] = col2;
				objects[col2] = row1;
				if (curRow != -1)
				{
					persons[curRow] = -1;
					bannedSwitches[row1 * numberOfObjects + curRow] = true;
				}
				//				if (pRes)
				//					printf("P%d O%d ->%f \n", row1, col2, myDiff.value);

			}
		}
	}
}

//void smartInitialAssignmentWithInitial(int initialAssignmet[])
//{
//	int row1, curRow, col2, i;
//	//  time_t start = time(NULL);
//	for (i = 0; i < numberOfPersons; i++)
//	{
//		if (initialAssignmet[i] != -1 && objects[initialAssignmet[i]] == -1
//				&& aijMatrix[i][initialAssignmet[i]] != negInf)
//		{
//			persons[i] = initialAssignmet[i];
//			objects[initialAssignmet[i]] = i;
//		} else
//		{
//			curRow = i;
//			while (curRow != -1)
//			{
//				Difference myDiff = getRowBestDifference(curRow);
//				if (myDiff.index != -1)
//				{
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
//					if (curRow != -1)
//					{
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


//void addRowBestDifference(int rowId, int sort)
//{
//	Difference myDifference = getRowBestDifference(rowId);
//	if (myDifference.index != -1)
//	{
//		myDifference.myAssigned = persons[rowId];
//		myDifference.bestChangeAssigned = objects[myDifference.bestChange];
//		if (sort == 0)
//		{
//			differences[rowId] = myDifference;
//		} else
//		{
//			addSortedDifference(myDifference);
//		}
//		rowDifferences[rowId] = myDifference;
//	}
//}

//JNIEXPORT jobjectArray JNICALL Java_se_peertv_opto_ImprovedHeuristic_JniHeuristic_heuristic
//(JNIEnv *env, jobject self,
// jobjectArray aijMatrixCp, int numberOfPersonsCp, int numberOfObjectsCp, jintArray initialAssignments) {
//	// for test ...
//	int C = 300;
//	numberOfPersons = 100;
//	numberOfObjects = 100;
//	emptyDiff.index = -1;
//	float *aijPtr;
//	aijPtr = malloc(numberOfPersons * numberOfObjects * sizeof(float));
//	int *bannedSwitchesPtr;
//	bannedSwitchesPtr = malloc(numberOfPersons * numberOfObjects * sizeof(int));
//	aijMatrix = malloc(numberOfPersons * sizeof(float *));
//	bannedSwitches = malloc(numberOfPersons * sizeof(int *));
//	int i,j;
//
//	for (i = 0; i < numberOfPersons; i++) {
//		aijMatrix[i] = aijPtr + (i*numberOfObjects);
//		bannedSwitches[i] = bannedSwitchesPtr + (i*numberOfObjects);
//	}
//
//	for (i = 0; i < numberOfPersons; i++) {
//		for (j = 0; j < numberOfObjects; j++) {
//			aijMatrix[i][j] = random() % C;
//			//printf("j has the value %f and is stored at %p\n", aijMatrix[i][j], (void *)&aijMatrix[i][j]);
//			//printf("%f jjj ",aijMatrix[i][j]);
//		}
//	}s
//	int *initialAssignment;
//	initialAssignment = malloc(numberOfPersons * sizeof(int));
//	int len = 0;
//	if (len > 0) {
//		for(i = 0; i < numberOfPersons; i++)
//		{
//			initialAssignment[i] = i;
//		}
//	}
//	clearedBannedSwitches = malloc(numberOfPersons * sizeof(int));
//	persons = malloc(numberOfPersons * sizeof(int));
//	differences = malloc((numberOfPersons + numberOfObjects) * sizeof(Difference));
//	rowDifferences = malloc(numberOfPersons * sizeof(Difference));
//	for (i = 0; i < numberOfPersons; i++) {
//		clearedBannedSwitches[i] = 0;
//		persons[i] = -1;
//		for (j = 0; j < numberOfPersons; j++)
//			bannedSwitches[i][j] = -1;
//		rowDifferences[i] = emptyDiff;
//		differences[i] = emptyDiff;
//	}
//
//	columnDifferences = malloc(numberOfObjects * sizeof(Difference));
//	objects = malloc(numberOfObjects * sizeof (int));
//	for (i = 0; i < numberOfObjects; i++) {
//		objects[i] = -1;
//		columnDifferences[i] = emptyDiff;
//		differences[numberOfPersons + i] = emptyDiff;
//	}
//
//	if (len > 0)
//		smartInitialAssignmentWithInitial(initialAssignment);
//	else
//		smartInitialAssignment();
//	enhanceBySwitching();
//	for (i = 0; i < numberOfPersons; i++) {
//		printf("aij ");
//		printf("%d ",i);
//		printf("%d ",persons[i]);
//		printf("%f \n",aijMatrix[i][persons[i]]);
//	}
//	return 0;
//	jintArray r= (*env)->NewIntArray(env, numberOfPersons);
//	//for (i = 0; i < numberOfPersons; i++)
//	//	(*env)->SetObjectArrayElement(env, r,i,persons[i]);
//	return r;
//}


void printH(float** aijMatrix, int numberOfPersons, int numberOfObjects) {
	for (int i = 0; i < numberOfPersons; i++)
	{
		for (int j = 0; j < numberOfObjects; j++)
		{
			printf("%f ", aijMatrix[i][j]);
			//printf("j has the value %f and is stored at %p\n", aijMatrix[i][j], (void *)&aijMatrix[i][j]);
			//printf("%f jjj ",aijMatrix[i][j]);
		}
		printf("\n");
	}
}

void printG(float* h_A, int numberOfPersons, int numberOfObjects) {
	for (int i = 0; i < numberOfPersons; i++)
	{
		for (int j = 0; j < numberOfObjects; j++)
		{
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
	for (dev = 0; dev < deviceCount; ++dev)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		if (dev == 0)
		{
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
		printf("  Total amount of global memory:                 %u bytes\n",
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
	if (cudaSuccess != err)
	{
		fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(-1);
	}
}
