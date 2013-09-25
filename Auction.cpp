/*
 * auction.cpp
 *
 *  Created on: Dec 28, 2009
 *      Author: amgadnaiem
 */

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <vector>
#include <limits>



using namespace std;

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



class Auction {

    public:
         int numberOfPersons , numberOfObjects;
         float e ; //the E-Complementary Slackness parameter

         int *persons;
        //this vector will be containing the instances of class Person
         int *objects;
        //this vector will be containing the instances of class Object
         float *prices;
        //this represents the minimum price that must be paid in order to get an object
         float **aijMatrix;
        //this matrix represents the prices that persons are willing to pay to each object
         bid* bids;

           Auction(int numObjects, int numPersons, float** aij){
        // constructor will be responsible of initializing the AuctioningEngine ... Creating Instances... assigning them to the arrays etc...

        numberOfObjects = numObjects;
        numberOfPersons = numPersons;
        e = input.e;

        persons = new int[numberOfPersons];
        objects = new int[numberOfObjects];
        aijMatrix = aij;
        bids = new bid[numberOfObjects];
        prices = new float [numberOfObjects];

        //the following loop creates persons = numberOfPersons and add them to the persons vector
        //it also initializes the unassigned Set where all persons are unassigned and the assignmentVector where all persons are not assigned to any object
        for (int i = 0; i < numberOfPersons; i++){
            persons[i] = -1;
        }

        //the following loop creates unassigned objects (with everything = -1) and add them to the objects vector
        for (int i = 0; i < numberOfObjects; i++){
            objects[i] = -1;
            bids[i].value = -1;
            prices[i] = 0;
            bids[i].winner = -1;
        }
    };
      ~Auction(){

        delete[] persons;
        delete[] objects;
        delete[] aijMatrix;
        delete[] bids;
        delete[] prices;
    };
	       void startBiddingPhase(){
        /*this method is responsible of starting the bidding phase where we determine the best object's value for each person
        and the second best value and calculates the increments and sets the bids matrix that will be used in the assignment phase */

        //the following loop determines the best and second best for each person
        float max = 0;
        for (int i = 0; i < numberOfPersons; i++ ){
        	int best = -1;
        	int secondBest = -1;
        	float value = -1;
        	float secondBestValue = -10000;
        	if (persons[i] == -1){
        		max = -numeric_limits<float>::max();
	        	for(int j = 0 ; j < numberOfObjects; j++){
	        		value = aijMatrix[i][j] - prices[j];
	        		if(value >= max){
	        			if (max != 0)
	        			{
	        				secondBestValue = max;
	        			}
	        			secondBest = best;
	        			best = j;
	        			max = value;
	        		}
	        	}

	        if	(secondBest == -1)
	        {
	        	secondBestValue = -numeric_limits<float>::max();
	        	for(int j = 1 ; j < numberOfObjects; j++){
	        		value = aijMatrix[i][j] - prices[j];
	        		if(value > secondBestValue){
	        			secondBest = j;
	        			secondBestValue = value;
	        		}
	        	}
	        }
	        // from the calculation of the best and secondBest bids are calculated
	        float thisBid = prices[best] +  max - secondBestValue + e;

			//add the calculated bid to the Bids array if there is no higher bid is made
        	if (thisBid > bids[best].value) {

        		bids[best].value = thisBid;
        		bids[best].winner = i;

        	}

        	}
        }
     };
	       void startAssignmentPhase(){
    	/*this method is responsible of starting the assignment phase where the highest bid for each object is determined and the associated person
    	 is assigned to that object */

    	for(int j = 0 ; j < numberOfObjects ; j++){
        	//the following condition to prevent the bidder to assigned to more than one object
        	if (bids[j].winner != -1 && persons[bids[j].winner] == -1){

        		if(objects[j] != -1)
        			persons[objects[j]] = -1;

        		objects[j] = bids[j].winner;
        		persons[bids[j].winner] = j;
        		prices[j] = bids[j].value;

        	}

        }
    };

	void startAuctioning(){
        //this method is responsible of executing the auctioning algorithm by calling the startBiddingPhase then the startAssignmentPhase

       bool allSatisfied = false;
       int counter = 0;

       //this is the main loop that will iterate till all persons are Satisfied i.e. each is assigned to an object
       while(allSatisfied == false){
        	counter++;
        	//cout << "start " <<counter << endl;
        	for ( int i = 0; i < numberOfObjects; i++)
        	{
        		bids[i].value = -1;
        		bids[i].winner = -1;
        	}
        	//cout << "before bid " <<counter << endl;
    	    startBiddingPhase();
			startAssignmentPhase();
        	allSatisfied = true;
	        for (int i = 0 ;i < numberOfPersons; i++)
        	{
        		if(persons[i] == -1 ){
        			allSatisfied = false;
        			break;
        		}
        	}
    	}
	};

	   float calculateTotalHappiness() {
		float totalHappiness = 0;
		int i;
		for (i = 0; i < numberOfPersons; i++) {
			if (persons[i] != -1)
				totalHappiness += aijMatrix[i][persons[i]];
		}
		return totalHappiness;
	   };
};

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

int main( int argc, const char* argv[] )
{
	float** aijMatrix = genMatrix(16, 16, 1);

	Auction a=Auction(16, 6,aijMatrix);
	a::startAuctioning();
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

