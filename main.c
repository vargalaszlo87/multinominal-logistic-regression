/**
* @project multinominal-logistic-regression v.0.9
* @file main.c
* @date 2024-04-25
* @author Laszlo, Varga <varga.laszlo.drz@gmail.com> vargalaszlo.com
* @brief This is the "how to use" file.
*/

#include <stdio.h>
#include <stdlib.h>

#include "include/multilogreg.h"

int main()
{
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
}
