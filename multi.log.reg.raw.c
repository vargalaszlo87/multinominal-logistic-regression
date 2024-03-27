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
    double *y;
    double *weight;
    int sample;
    int feature;
    int xyCounter;
    int wCounter;
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
    (d->weight+d->wCounter) = w;

    d->wCounter++;
}

/*
bool multiLogRTrain() {
    return true;
}
*/

double dot_product(double *v1, double *v2, int size) {
    double result = 0.0;
    for (int i = 0; i < size; ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

void train(double X[NUM_SAMPLES][NUM_FEATURES], int y[NUM_SAMPLES], double weights[NUM_FEATURES], int num_iterations) {
    for (int iter = 0; iter < num_iterations; ++iter) {
        for (int i = 0; i < NUM_SAMPLES; ++i) {
            double prediction = sigmoid(dot_product(weights, X[i], NUM_FEATURES));
            for (int j = 0; j < NUM_FEATURES; ++j) {
                weights[j] += LEARNING_RATE * (y[i] - prediction) * X[i][j];
            }
        }
    }
}

int predict(double x[NUM_FEATURES], double weights[NUM_FEATURES]) {
    double prediction = sigmoid(dot_product(weights, x, NUM_FEATURES));
    if (prediction >= 0.5) {
        return 1;
    } else {
        return 0;
    }
}

int main() {
    double X[NUM_SAMPLES][NUM_FEATURES] = {{2.0, 1.0, 1.0},
                                           {3.0, 2.0, 0.0},
                                           {3.0, 4.0, 0.0},
                                           {5.0, 5.0, 1.0},
                                           {7.0, 5.0, 1.0},
                                           {2.0, 5.0, 1.0}};
    int Y[NUM_SAMPLES] = {0, 0, 0, 1, 1, 1}; // Osztálycímkék


    double weights[NUM_FEATURES] = {0.0}; // Súlyok inicializálása


    Data data;
    Result result;

    data.sample = NUM_SAMPLES;
    data.feature = NUM_FEATURES;

    multiLogRInit(&data, &result);

    for (int i = 0; i < NUM_SAMPLES ; i++)
        multiLogRPush(&data, X[i], Y[i]);

    for (int i = 0; i < NUM_FEATURES ; i++)
        multiLogRPushWeight(&data, weights[i]);


    printf ("%lf", data.x[0][2]);

/*
    // Modell tanítása
    train(X, y, weights, MAX_ITERATIONS);

    // Tesztadatok
    double test_data[NUM_FEATURES] = {7.2, 5.0, 1.0};

    // Predikció
    int prediction = predict(test_data, weights);
    printf("Prediction: %d\n", prediction);
*/
    return 0;
}
