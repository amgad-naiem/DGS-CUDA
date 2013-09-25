/*
 *  Generator.cpp
 *  heuristic C
 *
 *  Created by Amgad Naiem on 9/7/09.
 *  Copyright 2009 Peerialism. All rights reserved.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Global.h"

int** square; // representing the 2D square
int* listOfLocX; //for each element placed in the 2D square we save it's location on x (i index of the 2D square(2d array)) to not search for it again when calculating the euclidian distance...the index represent the object
int* listOfLocY; //for each element placed in the 2D square we save it's location on y (j index of the 2D square(2d array)) to not search for it again when calculating the euclidian distance...the index represent the object
int numberOfPersonsGen;
int numberOfObjectsGen;
float** aijMatrixGen;
bool* chosenIndeces;
int c; //parameter to adjust the dimension of the 2D square used as space for the Euclidian distance between each 2 points
void generate();
float getEuclidianDistance(int, int);

/**
 * The class constructor
 *
 * @param numberOfPersons of type integer
 * @param numberOfObjects of type integer
 */
//int main() {
//	numberOfObjectsGen = 16;
//	numberOfPersonsGen = 16;
//	c = 300;
//	srand(7);
//	aijMatrixGen = new float*[numberOfPersonsGen];
//	for (int i = 0; i < numberOfPersonsGen; i++)
//		aijMatrixGen[i] = new float[numberOfObjectsGen];
//	chosenIndeces = new bool[numberOfObjectsGen];
//	listOfLocX = new int[numberOfObjectsGen];
//	listOfLocY = new int[numberOfObjectsGen];
//	square = new int*[c];
//	for (int i = 0; i < c; i++)
//	{
//		square[i] = new int[c];
//	}
//
//	//TODO: check that c*c >= numberOfObjectsGen
//
//	//initializing all the square by -1 as there will be empty places ... because after placing objects (persons) indeces there will be empty places
//	//recognized by the value of -1
//	for (int i = 0; i < c; i++)
//	{
//		for (int j = 0; j < c; j++)
//		{
//			square[i][j] = -1;
//		}
//	}
//
//	for (int i = 0; i < numberOfObjectsGen; i++)
//	{
//		chosenIndeces[i] = false;
//	}
//
//	generate();
//
//	for (int i = 0; i < numberOfPersonsGen; i++)
//	{
//		for (int j = 0; j < numberOfObjectsGen; j++)
//		{
//			printf("%f ,", aijMatrixGen[i][j]);
//		}
//		printf("\n");
//	}
//
//	return 0;
//
//}

float** genMatrix(int numberOfPersonsPassed, int numberOfObjectsPassed, int seed) {
	numberOfPersonsGen=numberOfPersonsPassed;
	numberOfObjectsGen=numberOfObjectsPassed;
	c = 300;
	srand(seed);
	aijMatrixGen = new float*[numberOfPersonsGen];
	for (int i = 0; i < numberOfPersonsGen; i++)
		aijMatrixGen[i] = new float[numberOfObjectsGen];
	chosenIndeces = new bool[numberOfObjectsGen];
	listOfLocX = new int[numberOfObjectsGen];
	listOfLocY = new int[numberOfObjectsGen];
	square = new int*[c];
	for (int i = 0; i < c; i++)
	{
		square[i] = new int[c];
	}

	//TODO: check that c*c >= numberOfObjectsGen

	//initializing all the square by -1 as there will be empty places ... because after placing objects (persons) indeces there will be empty places
	//recognized by the value of -1
	for (int i = 0; i < c; i++)
	{
		for (int j = 0; j < c; j++)
		{
			square[i][j] = -1;
		}
	}

	for (int i = 0; i < numberOfObjectsGen; i++)
	{
		chosenIndeces[i] = false;
	}

	generate();

	return aijMatrixGen;

}

/**
 *
 * The method that generates the random numbers following the GEOM algorithm mentioned above
 * @return aijMatrixGen 2 dimensional array of type float
 */
void generate() {

	int numGenerated = 0;
	while (numGenerated < numberOfObjectsGen)
	{ //we are always concerned by the number of objects because it is the one that might be larger so that the indices
		// in the 2D square must cover all possible values of i,j of aijMatrixGen

		int randomIndex = rand() % numberOfObjectsGen;

		while (chosenIndeces[randomIndex] == true)
		{ // if it is not chosen before so we can add else we cannot add two indices with the same value in the 2D square
			randomIndex = rand() % numberOfObjectsGen;
		}

		chosenIndeces[randomIndex] = true;
		numGenerated++;

		int randomLocX = rand() % c; //coordinate x to place the random number generated in the 2D square
		int randomLocY = rand() % c; //coordinate y to place the random number generated in the 2D square

		while (square[randomLocX][randomLocY] != -1)
		{ //if the place is already taken choose another one
			randomLocX = rand() % c;
			randomLocX = rand() % c;
		}

		square[randomLocX][randomLocY] = randomIndex;
		listOfLocX[randomIndex] = randomLocX;
		listOfLocY[randomIndex] = randomLocY;
	}

	// Now we have initialized the 2D Square we need to calculate the aijMatrixGen values by calculating the Euclidian distance between i and j in the 2D Square

	for (int i = 0; i < numberOfPersonsGen; i++)
	{
		for (int j = 0; j < numberOfObjectsGen; j++)
		{
			aijMatrixGen[i][j] = floor(getEuclidianDistance(i, j) + 0.5); // this element of aijMatrixGen equals the euclidian distance between the i,j in the 2D Square
		}
	}

	//return aijMatrixGen;
}

/**
 *
 * This method is responsible of calculating the euclidian distance in the 2D Square between the two points x , y .
 * @param x represents the first point needed to calculate the euclidian distance of type integer < numberOfObjectsGen
 * @param y represents the second point needed to calculate the euclidian distance of type integer < numberOfObjectsGen
 * @return the euclidian distance of type float
 */
float getEuclidianDistance(int point1, int point2) {
	float euclidianDistance = 0;
	int point1LocX = -1; //x coordinate of first point
	int point2LocX = -1; //x coordinate of second point

	int point1LocY = -1; // y coordiante of first point
	int point2LocY = -1; // y coordinate of second point


	if (point1 == point2)
		return euclidianDistance = 0; // same point ..then no need to calculate euclidian distance

	// first we need to now the location of each point the 2D square

//	int counter = 0; // to count the number of found squares

	point1LocX = listOfLocX[point1];
	point1LocY = listOfLocY[point1];

	point2LocX = listOfLocX[point2];
	point2LocY = listOfLocY[point2];

	// then we calculate the euclidian between the 2 points locations
	euclidianDistance = sqrt(pow(point1LocX - point2LocX, 2) + pow(point1LocY
			- point2LocY, 2));

	return euclidianDistance;
}
