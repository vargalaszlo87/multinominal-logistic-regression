# multinominal-logistic-regression

A multinomial logistic regression (or multinomial regression for short) is used when the outcome variable being predicted is nominal and has more than two categories that do not have a given rank or order. 

This tool has been written in pure C99 without dependencies. Optimization methods in this version are **Gradient Descent** and **Stochastic Gradient Descent**.

## Setup

The default value is the bold parameter.

### Basic

`learningRate` - **0.01** is the learning rate of algorithm.

`maxIteration` - **10000** is the max iteraton number.

### Losses

`lossMethod` - **LOG_LOSS** / BINARY_CROSS_ENTROPY

### Regularization

`regularizationMethod` - **NONE** / L1 / L2

`lambda` - **0.05** - Regularization parameter for method.

### Early-Stop

`earlyStopMethod` - **NONE** / LOSS_LIMIT / ACCURACY_LIMIT / LOSS_PATIENCE / ACCURACY_PATIENCE

`earlyStopLimit` - **0.5** - If you use the LOSS or ACCURACY limit, this is the limit value. 

`earlyStopPatience` - **10** - If you use the LOSS or ACCURACY patience, the 10th stops at the same.

### Optimization

`optimizationMethod` - **GD** / SGD - _(Gradient Descent / Stochastic Gradient Descent)_


## Usage

Insert the header:
```C
#include "multilogreg.h"
```
Use the Data structure:
```C
Data data;
```
Test data (in this case hardcoded):
```C
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
```
Make an initialization
```C
multiLogRegInit(&data, SAMPLES, FEATURES);
```
Settings
```C
// basic
data.setup.maxIteration = 11000;
data.setup.learningRate = 0.02;

// regularization
data.setup.regularizationMethod = L1;
data.setup.lambda = 0.05;

// losses
data.setup.lossMethod = LOG_LOSS;

// early-stop
data.setup.earlyStopMethod = LOSS_PATIENCE;
data.setup.earlyStopPatience = 10;
data.setup.earlyStopEpsilon = 0.0001;
```
Push the datas
```C
for (int i = 0; i < data.setup.sampleSize ; i++)
    multiLogRegPushData(&data, X[i], Y[i]);
```

Set the weights
```C
// add the weights manually
//for (int i = 0; i < setup.featureSize ; i++)
//    multiLogRPushWeight(&data, weights[i]);
//
// add the weights automatic (from normal distribution)
multiLogRegAutoWeight(&data);
```

Train the model
```C
multiLogRegTrain(&data);
```

Check it out
```C
// check
double input_data[FEATURES] = {7.0, 5.0, 0.8};

// predection
data.result = multiLogRegPredict(&data, input_data);
```

Show the result
```C
// show the result
printf("The prediction is %d with the (%.1lf %.1lf %.1lf) input numbers.\n", data.result, input_data[0], input_data[1], input_data[2]);

printf ("Loss: %lf, Accuracy: %lf\n", data.loss, data.accuracy);

printf ("Iteration: %d", data.iteration);
```

## Self compiling

If you use **`Windows`**
```
gcc -std=c99 -c main.c -o build/main.o -I"include"
gcc -std=c99 -c src/multilogreg.c -o build/multilogreg.o -I"include"
gcc build/main.o build/multilogreg.o -o build/multilogreg.exe
```

If you use **`Linux`**
```
gcc -std=c99 -c main.c -o build/main.o -I"include" -lm
gcc -std=c99 -c src/multilogreg.c -o build/multilogreg.o -I"include" -lm
gcc build/main.o build/multilogreg.o -o build/multilogreg
```

## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/) License.

## Contact

Varga Laszlo - https://vargalaszlo.com - mail@vargalaszlo.com.com

Project Link: https://github.com/vargalaszlo87/multinominal-logistic-regression

[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](http://vargalaszlo.com)
