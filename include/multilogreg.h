/**
* @project multinominal-logistic-regression v.0.9
* @file multilogreg.h
* @date 2024-04-25
* @author Laszlo, Varga <varga.laszlo.drz@gmail.com> vargalaszlo.com
* @brief header file
*/

#ifndef MULTILOGREG_H_INCLUDED
#define MULTILOGREG_H_INCLUDED

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <float.h>
#include <limits.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626463383279
#endif

// data structures

typedef struct Setup {
	enum none {
		NONE = 0
	};
    double learningRate;
    unsigned int maxIteration;
    unsigned int sampleSize;
    unsigned int featureSize;
    // optimization
    unsigned short optimizationMethod;
    enum optimization {
    	GD = 0,		// Gradient Descent
    	SGD = 1,	// Stochastic Gradient Descent
	};
    // early stop
    unsigned short earlyStopMethod;
    enum early {
    	// NONE
        LOSS_LIMIT = 1,
        ACCURACY_LIMIT = 2,
        LOSS_PATIENCE = 3,
        ACCURACY_PATIENCE = 4
    };
    double earlyStopValue;
    double earlyStopPatience;
    double earlyStopEpsilon;
    // regularization
    double lambda;
    unsigned short regularizationMethod;
    enum regularization {
		// NONE
        L1 = 1, // Lasso regression
        L2 = 2  // Ridge regression
    };
    // loss
    unsigned short lossMethod;
    enum loss {
        LOG_LOSS,
        BINARY_CROSS_ENTROPY
    };

} Setup;

typedef struct Stack {
		int *array;
		int size;
		int pointer;
} Stack;

typedef struct XY {
    double **x;
    unsigned int *y;
	unsigned int counter;
} XY;

typedef struct W {
    double *w;
    unsigned int counter;
} W;

typedef struct Data  {
	XY training;
 	XY validation;
 	W weight;
	Stack validationIndex;
    Setup setup;
    double accuracy;
    double loss;
    int iteration;
    int result;
} Data, *pData;

// methods

// auxiliary
void push(Stack * , int);
int pop(Stack *);
bool search(Stack* , int);
double calcDotProduct(double *, double *, unsigned int);
double calcSigmoid(double);
double calcBoXMuller();
double doubleRound(double , int);

// core
bool multiLogRegInit(Data *, unsigned int, unsigned int);
bool multiLogRegPushData(Data *, double *, double);
bool multiLogRegPushWeight(Data *, double );
bool multiLogRegAutoWeight(Data *);
bool multiLogRegTrain(Data *);
int multiLogRegPredict(Data *, double *);

#endif // MULTILOGREG_H_INCLUDED
