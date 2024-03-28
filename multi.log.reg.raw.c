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

typedef struct Data  {
    double **x;
    unsigned int *y;
    double *weight;
    unsigned int sample;
    unsigned int feature;
    unsigned int xyCounter;
    unsigned int wCounter;
} Data, *pData;

typedef struct Result {
    int sample;
} Result, *pResult;

bool multiLogRInit(Data *d, Result *r) {
    if (d->sample < 1 || d->feature < 1)
        return false;
    d->x = (double**)calloc(d->sample, sizeof(double*) + d->feature * sizeof(double));
    if (!d -> x)
            return false;
    for (int i = 0 ; i < d->sample ; ++i)
        *(d->x+i) = (double*)(d->x + d->sample) + i + d->feature;
    d->y = (int *)calloc(d->sample, sizeof(int));
    d->weight = (double *)calloc(d->feature, sizeof(double));
    if (!d -> y || !d -> weight)
        return false;
    d->xyCounter = 0;
    d->wCounter = 0;
    r->sample = d->sample;
}

bool multiLogRPush(Data *d, double *x, double y) {
    d->x[d->xyCounter] = x;
    d->y[d->xyCounter] = y;
    d->xyCounter++;
    return true;
}

bool multiLogRPushWeight(Data *d, double w) {
    if (d->feature == d->wCounter)
        return false;
    *(d->weight+d->wCounter) = 0.0;
    d->wCounter++;
}

double calcDotProduct(double *num1, double *num2, unsigned int size) {
    double r = 0.0;
    for (int i = 0; i < size; ++i) {
        r += num1[i] * num1[i];
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


void train(double X[NUM_SAMPLES][NUM_FEATURES], int y[NUM_SAMPLES], double weights[NUM_FEATURES], int num_iterations) {
    for (int iter = 0; iter < num_iterations; ++iter) {
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            double prediction = calcSigmoid(calcDotProduct(weights, X[i], NUM_FEATURES));
            for (int j = 0; j < NUM_FEATURES; ++j) {
                weights[j] += LEARNING_RATE * (y[i] - prediction) * X[i][j];
            }
        }
    }
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

    data.sample = NUM_SAMPLES;
    data.feature = NUM_FEATURES;


    // init
    multiLogRInit(&data, &result);

    // push datas
    for (int i = 0; i < NUM_SAMPLES ; i++)
        multiLogRPush(&data, X[i], Y[i]);

    for (int i = 0; i < NUM_FEATURES ; i++)
        multiLogRPushWeight(&data, weights[i]);

    // Modell tanítása
    train(X, Y, weights, MAX_ITERATIONS);

    // Tesztadatok
    double test_data[NUM_FEATURES] = {7.2, 5.0, 1.0};

    // Predikció
    int prediction = predict(test_data, weights);
    printf("Prediction: %d\n", prediction);

    return 0;
}
