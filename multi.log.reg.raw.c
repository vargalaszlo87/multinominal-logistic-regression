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

// auxiliary functions

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

double doubleRound(double value, int d) {
    double temp;
    double shift = pow(10, d);
    temp = value * shift;
    temp = round(temp);
    temp /= shift;
    return temp;
}

// core functions

bool multiLogRegInit(Data *d, unsigned int samples, unsigned int features) {
    if (samples < 1 || features < 1)
        return false;
    d->setup.sampleSize = samples;
	d->setup.featureSize = features;
    d->training.x = (double**)calloc(d->setup.sampleSize, sizeof(double*) + d->setup.featureSize * sizeof(double));
    if (!d -> training.x)
            return false;
    for (int i = 0 ; i < d->setup.sampleSize ; ++i)
        *(d->training.x+i) = (double*)(d->training.x + d->setup.sampleSize) + i + d->setup.featureSize;
    d->training.y = (int *)calloc(d->setup.sampleSize, sizeof(int));
    d->weight.w = (double *)calloc(d->setup.featureSize, sizeof(double));
    if (!d -> training.y || !d -> weight.w)
        return false;
    d->training.counter = 0;
    d->weight.counter = 0;
    // default
    d->setup.optimizationMethod = GD;
    d->setup.lossMethod = LOG_LOSS;
    d->setup.regularizationMethod = NONE;
    d->setup.lambda = 0.05;
    d->setup.earlyStopMethod = NONE;
    d->setup.earlyStopValue = 0.5;
    d->setup.earlyStopPatience = 10;
    d->iteration = 0;
    return true;
}

bool multiLogRegPushData(Data *d, double *x, double y) {
	if (d->training.counter == d->setup.sampleSize)
		return false;
    d->training.x[d->training.counter] = x;
    d->training.y[d->training.counter++] = y;
    return true;
}

bool multiLogRegPushWeight(Data *d, double w) {
	if (d->weight.counter == d->setup.featureSize)
		return false;
    *(d->weight.w + d->weight.counter++) = w;
    return true;
}

bool multiLogRegAutoWeight(Data *d) {
	srand(time(NULL));
	for (int i = 0 ; i < d->setup.featureSize ; i++) {
		*(d->weight.w + i) = calcBoXMuller();
	}
}

bool multiLogRegTrain(Data *d) {
	int iteration;
   	double earlyStopLossTemp = DBL_MAX;
   	double earlyStopAccuracyTemp = DBL_MIN;
   	double earlyStopPatienceTemp = d->setup.earlyStopPatience;
    for (iteration = 0 ; iteration < d->setup.maxIteration ; ++iteration) {
    	unsigned int correctPredictions = 0;
    	double totalLoss = 0.0;

		// optimization
		if (d->setup.optimizationMethod == SGD) {
            for (int i = d->training.counter - 1; i > 0; --i) {
                int j = rand() % (i + 1);
                // Swap data
                double *tempX = d->training.x[i];
                d->training.x[i] = d->training.x[j];
                d->training.x[j] = tempX;
                int tempY = d->training.y[i];
                d->training.y[i] = d->training.y[j];
                d->training.y[j] = tempY;
            }			
		}

        for (int i = 0; i < d->training.counter; ++i) {
            // sigmoid
            double p = calcSigmoid(calcDotProduct(d->weight.w, d->training.x[i], d->setup.featureSize));
            // accuracy
			int pClass = (p >= 0.5) ? 1 : 0;
            if (pClass == d->training.y[i]) {
                correctPredictions++;
            }
            // loss
            double loss;
            switch (d->setup.lossMethod) {
                case LOG_LOSS:
                   loss = -((d->training.y[i]) * log(p) + (1 - d->training.y[i]) * log(1 - p));
                   break;
                case BINARY_CROSS_ENTROPY:
                    loss = -(d->training.y[i]) * log(p) - (1 - d->training.y[i]) * log(1 - p);
                    break;
                default:
                    loss = 0;
                    break;
            }
            totalLoss += loss;
            // train + regularization
            for (int j = 0; j < d->setup.featureSize; ++j) {
                double gradient = (d->training.y[i] - p) * d->training.x[i][j];
                switch (d->setup.regularizationMethod) {
                    L1:
                        *(d->weight.w+j) += d->setup.learningRate * (gradient - d->setup.lambda * abs(*(d->weight.w+j)));
                        break;
                    L2:
                        *(d->weight.w+j) += d->setup.learningRate * (gradient - d->setup.lambda * pow(*(d->weight.w+j),2));
                        break;
                    // NONE:
                    default:
                        *(d->weight.w+j) += d->setup.learningRate * gradient;
                        break;
                }
            }
        }
        d->accuracy = (double)correctPredictions / d->training.counter;
        d->loss = totalLoss/d->training.counter;

        // early stop
        // limit
        if (d->setup.earlyStopMethod == LOSS_LIMIT && d->loss <= d->setup.earlyStopValue)
            break;
        if (d->setup.earlyStopMethod == ACCURACY_LIMIT && d->accuracy >= d->setup.earlyStopValue)
            break;
        // patience
        if (d->setup.earlyStopMethod > 2) {
            bool condition;
            switch(d->setup.earlyStopMethod) {
                case LOSS_PATIENCE:
                    condition = doubleRound(d->loss,6) < earlyStopLossTemp;
                    break;
                case ACCURACY_PATIENCE:
                    condition = doubleRound(d->accuracy,6) > earlyStopAccuracyTemp;
                    break;
                default:
                    break;
            }
            if (condition) {
                switch(d->setup.earlyStopMethod) {
                case LOSS_PATIENCE:
                    earlyStopLossTemp = doubleRound(d->loss,6);
                    break;
                case ACCURACY_PATIENCE:
                    earlyStopAccuracyTemp = doubleRound(d->accuracy,6);
                    break;
                default:
                    break;
                }
                earlyStopPatienceTemp = d->setup.earlyStopPatience;
            }
            else {
                earlyStopPatienceTemp--;
            }
            if (earlyStopPatienceTemp == 0) {
                break;

            }
        }

	}
	d->iteration = iteration;
    return true;
}

int multiLogRegPredict(Data *d, double *inputData) {
    double p = calcSigmoid(calcDotProduct(d->weight.w, inputData, d->setup.featureSize));
    return (p >= 0.5) ? 1 : 0;
}

// usage

int main() {

	// structure
    Data data;

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

    // init
    multiLogRegInit(&data, SAMPLES, FEATURES);

	// setup
	// basic
	data.setup.maxIteration = 100000;
	data.setup.learningRate = 0.01;

	// regularization
	data.setup.regularizationMethod = L1;
	data.setup.lambda = 0.05;

	// losses
	data.setup.lossMethod = LOG_LOSS;

	// early-stop
	data.setup.earlyStopMethod = LOSS_PATIENCE;
	data.setup.earlyStopPatience = 10;
	data.setup.earlyStopEpsilon = 0.0001;

    // push (training) datas
    for (int i = 0; i < data.setup.sampleSize ; i++)
        multiLogRegPushData(&data, X[i], Y[i]);

    // weights
    //
    // add the weights manually
    //for (int i = 0; i < setup.featureSize ; i++)
    //    multiLogRPushWeight(&data, weights[i]);
    //
    // add the weights automatic (from normal distribution)
    multiLogRegAutoWeight(&data);

    // train
    multiLogRegTrain(&data);

    // check
    double input_data[FEATURES] = {7.0, 5.0, 0.8};

    // predection
    data.result = multiLogRegPredict(&data, input_data);

    // show the result
    printf("The prediction is %d with the (%.1lf %.1lf %.1lf) input numbers.\n", data.result, input_data[0], input_data[1], input_data[2]);

	printf ("Loss: %lf, Accuracy: %lf\n", data.loss, data.accuracy);

	printf ("Iteration: %d", data.iteration);
    return 0;
}
