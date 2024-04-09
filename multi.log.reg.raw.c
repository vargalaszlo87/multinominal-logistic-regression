#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.1415926535897932384626463383279
#endif

typedef struct Setup {
    double learningRate;
    unsigned int maxIteration;
    unsigned int sampleSize;
    unsigned int featureSize; 
} Setup, *pSetup;

typedef struct Stack {
		int *array;
		int size;
		int pointer;
} Stack;

typedef struct Data  {
    double **x;
    unsigned int *y;
    double *weight;
    unsigned int sampleCounter;
    unsigned int weightCounter;
    Stack validationIndex;
    Setup setup;
} Data, *pData;

typedef struct Result {
    int result;
} Result, *pResult;

// stack

void push(Stack* s, int item) {
	if (s->pointer < s->size-1) 
		s->array[++s->pointer] = item;
}

int pop(Stack* s) {
	return (double)((s->pointer == -1 ) ? -1 : s->array[s->pointer--]);
}

bool search(Stack* s, int item) {
	for (int i = 0; i < s->pointer + 2; i++)
		if (s->array[i] == item)
			return true;
	return false;
}

// calc

double calcDotProduct(double *n1, double *n2, unsigned int size) {
    double r = 0.0;
    for (int i = 0; i < size; ++i) {
        r += n1[i] * n2[i];
    }
    return r;
}

double calcSigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double calcBoXMuller() {
	double i = rand() / (RAND_MAX + 1.0);
	double j = rand() / (RAND_MAX + 1.0);
	return sqrt(-2 * log(i)) * cos(2 * M_PI * j);
}

// interface

bool multiLogRegInit(Data *d, Result *r) {
    if (d->setup.sampleSize < 1 || d->setup.featureSize < 1)
        return false;
    d->x = (double**)calloc(d->setup.sampleSize, sizeof(double*) + d->setup.featureSize * sizeof(double));
    if (!d -> x)
            return false;
    for (int i = 0 ; i < d->setup.sampleSize ; ++i)
        *(d->x+i) = (double*)(d->x + d->setup.sampleSize) + i + d->setup.featureSize;
    d->y = (int *)calloc(d->setup.sampleSize, sizeof(int));
    d->weight = (double *)calloc(d->setup.featureSize, sizeof(double));
    if (!d -> y || !d -> weight)
        return false;
    d->sampleCounter = 0;
    d->weightCounter = 0;
    return true;
}

bool multiLogRegPushData(Data *d, double *x, double y) {
	if (d->sampleCounter == d->setup.sampleSize)
		return false;
    d->x[d->sampleCounter] = x;
    d->y[d->sampleCounter++] = y;
    return true;
}

bool multiLogRegPushWeight(Data *d, double w) {
	if (d->weightCounter == d->setup.featureSize)
		return false;
    *(d->weight+d -> weightCounter++) = w;
    return true;
}

bool multiLogRegAutoWeight(Data *d) {
	srand(time(NULL));
	for (int i = 0 ; i < d->setup.featureSize ; i++) {
		*(d->weight + i) = calcBoXMuller();
	}
}

bool multiLogRegTrain(Data *d) {
    for (int iteration = 0 ; iteration < d->setup.maxIteration ; ++iteration)
        for (int i = 0; i < d->sampleCounter; ++i) {
            double p = calcSigmoid(calcDotProduct(d->weight, d->x[i], d->setup.featureSize));
            for (int j = 0; j < d->setup.featureSize; ++j) {
                *(d->weight+j) += d->setup.learningRate * (d->y[i] - p) * d->x[i][j];
            }
        }
    return true;
}

int multiLogRegPredict(Data *d, double *inputData) {
    double p = calcSigmoid(calcDotProduct(d->weight, inputData, d->setup.featureSize));
    return (p >= 0.5) ? 1 : 0;
}

bool multiLogRegMakeValidArray(Data * d, double ratio) {
	
	/*
	** DEV
	*/
	
	int div = floor(d->setup.sampleSize * ratio);
	if (ratio < 0.01 || ratio > 9.99 || div < 1)
		return false;
	// stack
	d->validationIndex.array = (int*)calloc(div, sizeof(int));
	d->validationIndex.pointer = -1;
	d->validationIndex.size = div;
	// core	
	srand(time(NULL));
	for (int i = 0; i < div ;i++) {
		int temp = rand() % (d->setup.sampleSize - 1 );
		if (search(&d->validationIndex, temp)) {
			i--;
			continue;
		}
		push(&d->validationIndex, temp);		
	}
}

int main() {

	// structure
    Data data;
    //Setup setup;
    Result result;

    // test datas  
    #define SAMPLES 6
    #define FEATURES 3
    
    double X[SAMPLES][FEATURES] = 
		{{2.0, 1.0, 1.0},
         {3.0, 2.0, 0.0},
         {3.0, 4.0, 0.0},
         {5.0, 5.0, 1.0},
         {7.0, 5.0, 1.0},
         {2.0, 5.0, 1.0}
		};

    int Y[SAMPLES] =
		{0, 0, 0, 1, 1, 1}; 
			
    double weights[FEATURES] =
		{0.0,0.0,0.0}; 

	// setup
    data.setup.maxIteration = 10000;
    data.setup.learningRate = 0.01;

    data.setup.sampleSize = SAMPLES;
    data.setup.featureSize = FEATURES;

    // init
    multiLogRegInit(&data, &result);

    // push (training) datas
    for (int i = 0; i < data.setup.sampleSize ; i++)
        multiLogRegPushData(&data, X[i], Y[i]);

    //for (int i = 0; i < setup.featureSize ; i++)
    //    multiLogRPushWeight(&data, weights[i]);
        
    multiLogRegAutoWeight(&data);

    // train
    multiLogRegTrain(&data);

    // check
    double input_data[FEATURES] = {2.2, 4.9, 1.8};

    // predection
    result.result = multiLogRegPredict(&data, input_data);
    
    // show the result
    printf("The prediction is %d with the (%.1lf %.1lf %.1lf) input numbers.\n", result.result, input_data[0], input_data[1], input_data[2]);

	multiLogRegMakeValidArray(&data, 0.48);
	
    return 0;
}
