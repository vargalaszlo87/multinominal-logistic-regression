#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

typedef struct Setup {
    double learningRate;
    unsigned int maxIteration;
    unsigned int sampleSize;
    unsigned int featureSize;
} Setup, *pSetup;

typedef struct Data  {
    double **x;
    unsigned int *y;
    double *weight;
    unsigned int sampleCounter;
    unsigned int wCounter;
} Data, *pData;

typedef struct Result {
    int result;
} Result, *pResult;

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

bool multiLogRInit(Setup *s, Data *d, Result *r) {
    if (s->sampleSize < 1 || s->featureSize < 1)
        return false;
    d->x = (double**)calloc(s->sampleSize, sizeof(double*) + s->featureSize * sizeof(double));
    if (!d -> x)
            return false;
    for (int i = 0 ; i < s->sampleSize ; ++i)
        *(d->x+i) = (double*)(d->x + s->sampleSize) + i + s->featureSize;
    d->y = (int *)calloc(s->sampleSize, sizeof(int));
    d->weight = (double *)calloc(s->featureSize, sizeof(double));
    if (!d -> y || !d -> weight)
        return false;
    d->sampleCounter = 0;
    d->wCounter = 0;
}

bool multiLogRPush(Data *d, double *x, double y) {
    d->x[d->sampleCounter] = x;
    d->y[d->sampleCounter] = y;
    d->sampleCounter++;
    return true;
}

bool multiLogRPushWeight(Data *d, double w) {
    *(d->weight+d->wCounter) = w;
    d->wCounter++;
}

bool multiLogRTrain(Setup *s, Data *d) {
    for (int iteration = 0 ; iteration < s->maxIteration ; ++iteration)
        for (int i = 0; i < d->sampleCounter; ++i) {
            double p = calcSigmoid(calcDotProduct(d->weight, d->x[i], s->featureSize));
            for (int j = 0; j < s->featureSize; ++j) {
                *(d->weight+j) += s->learningRate * (d->y[i] - p) * d->x[i][j];
            }
        }
    return true;
}

int multiLogRPredict(Setup *s, Data *d, double *inputData) {
    double p = calcSigmoid(calcDotProduct(d->weight, inputData, s->featureSize));
    return (p >= 0.5) ? 1 : 0;
}

int main() {

    Data data;
    Setup setup;
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
    setup.maxIteration = 10000;
    setup.learningRate = 0.01;

    setup.sampleSize = SAMPLES;
    setup.featureSize = FEATURES;

    // init
    multiLogRInit(&setup, &data, &result);

    // push (training) datas
    for (int i = 0; i < setup.sampleSize ; i++)
        multiLogRPush(&data, X[i], Y[i]);

    for (int i = 0; i < setup.featureSize ; i++)
        multiLogRPushWeight(&data, weights[i]);

    // train
    multiLogRTrain(&setup, &data);

    // check
    double input_data[FEATURES] = {2.2, 4.9, 1.8};

    // predection
    result.result = multiLogRPredict(&setup, &data, input_data);
    
    // show the result
    printf("The prediction is %d with the (%.1lf %.1lf %.1lf) input numbers.\n", result.result, input_data[0], input_data[1], input_data[2]);

    return 0;
}
