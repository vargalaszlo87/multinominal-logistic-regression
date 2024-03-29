#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define LEARNING_RATE 0.01
#define MAX_ITERATIONS 10000
#define NUM_CLASSES 2
#define NUM_FEATURES 3
#define NUM_SAMPLES 6

/* multiLogR() */

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
    int test;
} Result, *pResult;

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
    //if (s->featureSize == d->wCounter)
    //    return false;
    *(d->weight+d->wCounter) = w;
    d->wCounter++;
}

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


/*
bool multiLogRTrain() {
    return true;
}
*/


void old_multiLogRTrain(double X[NUM_SAMPLES][NUM_FEATURES], int y[NUM_SAMPLES], double weights[NUM_FEATURES], int num_iterations) {
    for (int iter = 0; iter < num_iterations; ++iter) {

        for (int i = 0; i < NUM_SAMPLES; ++i) {

            double prediction = calcSigmoid(calcDotProduct(weights, X[i], NUM_FEATURES));

            for (int j = 0; j < NUM_FEATURES; ++j) {

                weights[j] += LEARNING_RATE * (y[i] - prediction) * X[i][j];

            }
        }
    }
    printf ("%lf\n",  weights[2]);
}


bool multiLogRTrain(Setup *s, Data *d) {
    for (int iteration = 0 ; iteration < s->maxIteration ; ++iteration)
        for (int i = 0; i < d->sampleCounter; ++i) {
            double p = calcSigmoid(calcDotProduct(d->weight, d->x[i], s->featureSize));
            for (int j = 0; j < s->featureSize; ++j) {
                *(d->weight+j) += s->learningRate * (d->y[i] - p) * d->x[i][j];
            }
        }
     printf ("%lf\n",  *(d->weight+2));


}


int predict(double x[NUM_FEATURES], double weights[NUM_FEATURES]) {
    double prediction = calcSigmoid(calcDotProduct(weights, x, NUM_FEATURES));
    if (prediction >= 0.5) {
        return 1;
    } else {
        return 0;
    }
}

int main() {

    Data data;
    Setup setup;
    Result result;

    // test datas
    double X[NUM_SAMPLES][NUM_FEATURES] = {{2.0, 1.0, 1.0},
                                           {3.0, 2.0, 0.0},
                                           {3.0, 4.0, 0.0},
                                           {5.0, 5.0, 1.0},
                                           {7.0, 5.0, 1.0},
                                           {2.0, 5.0, 1.0}};

    int Y[NUM_SAMPLES] = {0, 0, 0, 1, 1, 1}; // Osztálycímkék
    double weights[NUM_FEATURES] = {0.0,0.0,0.0}; // Súlyok inicializálása



    setup.maxIteration = 10000;
    setup.learningRate = 0.01;

    setup.sampleSize = NUM_SAMPLES;
    setup.featureSize = NUM_FEATURES;


    // init
    multiLogRInit(&setup, &data, &result);

    // push datas
    for (int i = 0; i < NUM_SAMPLES ; i++)
        multiLogRPush(&data, X[i], Y[i]);

    for (int i = 0; i < NUM_FEATURES ; i++)
        multiLogRPushWeight(&data, weights[i]);

    // Modell tanítása
    //old_multiLogRTrain(X, Y, weights, MAX_ITERATIONS);
    multiLogRTrain(&setup, &data);

    // Tesztadatok
    double test_data[NUM_FEATURES] = {1.2, 1.0, 1.0};

    // Predikció
    int prediction = predict(test_data, weights);
    printf("Prediction: %d\n", prediction);

    return 0;
}
