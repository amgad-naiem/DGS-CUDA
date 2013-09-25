/*
 * Difference.c
 *
 *  Created on: Jul 15, 2009
 *      Author: amgadnaiem
 */

typedef struct {
	float value;
	int type; //0 for person 1 for object
	int index; //row number if type is 0 , or column number is type is 1
	int bestChange; // colum to change to if type is 0(row), or row if type is 1
	int bestChangeAssigned;
	int myAssigned;
	float mRmC;
	float mRoC;
	float oRmC;
	float oRoC;
} Difference;
