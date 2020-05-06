/*
Deep neural network accelerated with CUDA.

C++11 & CUDA
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <string>
#include <fstream>
#include <cuchar>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>


using namespace std;

typedef unsigned char uchar;

/*
TODO
- Faster w/ CUDA - 260s baseline
- Template
- Object Oriented
- C++ & CUDA memory leaks
- Put memory on GPU and leave it there while training / testing
*/

template <typename T>
cudaError_t CUDA_1i1o(void(*func)(T*, const T*), T *b, const T *a, unsigned int size);
template <typename T>
cudaError_t CUDA_2i1o(void (*func)(T*, const T*, const T*), T *c, const T *a, const T *b, unsigned int size);

__global__ void mul(double *output, const double *in1, const double *in2) {
	/*
	Matrix multiplication.

	Parameters
	----------
	output: double*
		Output array.
	in1: double*
		Input array.
	in2: double*
		Input array.

	Effects
	-------
	Update output per element with results of in1 and in2 multiplication.
	*/
	int i = threadIdx.x;

	output[i] = in1[i] * in2[i];
}

__global__ void sigmoid(double *output, const double *input){
	/*
	Sigmoid activation function.

	Parameters
	----------
	output: double*
		Output array.
	input: double*
		Input array.

	Effects
	-------
	Apply sigmoid function to all inputs and update output.
	*/
	int i = threadIdx.x;

	output[i] = tanh(input[i]);
}

__global__ void dsigmoid(double *output, const double *input){
	/*
	Derivative of sigmoid activation function.

	Parameters
	----------
	output: double*
		Output array.
	input: double*
		Input array.

	Effects
	-------
	Apply sigmoid function to all inputs and update output.
	*/
	int i = threadIdx.x;

	output[i] = 1.0 - pow(input[i], 2);
}

// CONVERT
double *matmul(const double *x, double **mat, const unsigned int maty, const unsigned int matx) {
	/*
	Matrix multiplication

	Parameters
	----------
	x: double[n]
	mat: double[m, n]
	maty: int = m
	matx: int = n

	Returns
	-------
	double[m] Output of matrix multiplication.
	*/
	double *output = new double[matx];

	for (unsigned int i = 0; i < matx; i++) {
		double temp = 0;

		for (unsigned int j = 0; j < maty; j++) {
			temp += x[j] * mat[j][i];
		}

		output[i] = temp;
	}

	return output;
}



double cross_entropy_loss(double *real, const double target, const unsigned int n_outputs) {
	/*
	Cross entropy loss function.

	Parameters
	----------
	real: double[n]
		Approximate label.
	target: double[n]
		Expected label.
	n_outputs: uint = n
		Number of values in arrays.

	Returns
	-------
	Cross entropy loss.
	*/
	double total = 0;

	for (unsigned int j = 0; j < n_outputs; j++)
		total += exp(real[j]);

	return -real[(int)target] + log(total);
}

double *dcross_entropy_loss(const double *real, const double *target, const unsigned int n_outputs) {
	/*
	Derivative of cross entropy loss function.

	Parameters
	----------
	real: double[n]
		Approximate label.
	target: double[n]
		Expected label.
	n_outputs: uint = n
		Number of values in arrays.

	Returns
	-------
	Derivative of cross entropy loss.
	*/

	double *output = new double[n_outputs];

	for (unsigned int i = 0; i < n_outputs; i++)
		output[i] = (real[i] - target[i]) / n_outputs;

	return output;
}

double* onehot(double target, int N_CLASS) {
	/*
	Create onehot vector.

	Parameters
	----------
	target: double
		Value to convert.
	N_CLASS: int
		Number of classes.

	Returns
	-------
	double* Onehot vector.
	*/
	double *output = new double[N_CLASS];

	for (int i = 0; i < N_CLASS; i++) {
		if (i == target)
			output[i] = 1.;
		else
			output[i] = 0.;
	}

	return output;
}

int amax(double *real, const unsigned int n_values) {
	/*
	Argmax function.

	Parameters
	----------
	real: double*
		Array to find argmax of.
	n_values: uint
		Number of values in real.

	Returns
	-------
	int Index of maximum value.
	*/
	unsigned int max_idx = n_values;
	double max = -9999;

	for (unsigned int i = 0; i < n_values; i++) {
		if (real[i] > max) {
			max = real[i];
			max_idx = i;
		}
	}

	return max_idx;
}

double **forward(double *x, double ***w, const unsigned int *layers, const unsigned int n_layers){
	/*
	Forward propogate input through network.

	Parameters
	----------
	x: double[n]
		Input vector.
	w: double[m, n]
		Weight matrix.
	layers: uint*
		Size of each layer in the network.
	n_layers: uint
		Number of layers

	Returns
	-------
	double** Fires in each layer of the network.
	*/
	double **fires = new double*[n_layers];
	fires[0] = x;

	double *temp;
	for (unsigned int i = 0; i < n_layers - 1; i++) {
		temp = matmul(fires[i], w[i], layers[i], layers[i+1]);
	
		fires[i + 1] = new double[layers[i + 1]];
		CUDA_1i1o(sigmoid, fires[i+1], temp, layers[i + 1]);
	}

	return fires;
}

void backward(double ***w, const double *target, double **fires, const unsigned int *layers, const unsigned int n_layers, double learning_rate) {
	/*
	Cross entropy loss function.

	Parameters
	----------
	w: double[m, n]
		Weight matrix.
	target: double[n]
		Expected label.
	fires: double[n_layers, layers]
		Output of each layer in the network.
	layers: uint*
		Number of neurons in each layer.
	n_layers: uint
		Number of layers.
	learning_rate: double
		Learning rate of the network.

	Effects
	-------
	Update weight matrix.
	*/
	double **deltas = new double*[n_layers - 1];
	double *error; double *activation_prime; double *delta;

	//// Output layer
	error = dcross_entropy_loss(fires[n_layers - 1], target, layers[n_layers - 1]);

	activation_prime = new double[layers[n_layers - 1]];
	CUDA_1i1o(dsigmoid, activation_prime, fires[n_layers - 1], layers[n_layers - 1]);

	delta = new double[layers[n_layers - 1]];
	CUDA_2i1o(mul, delta, activation_prime, error, layers[n_layers - 1]);

	deltas[n_layers - 2] = delta;

	//// Backpropogate
	for (int k = n_layers - 3; k > -1; k--) {
		int kk = n_layers - 3 - k;

		delete[] error;
		error = new double[layers[n_layers - 2 - kk]];
		for (int i = 0; i < layers[n_layers - 2 - kk]; i++)
		{
			error[i] = 0;

			for (int j = 0; j < layers[n_layers - 1 - kk]; j++) {

				error[i] += w[k + 1][i][j] * deltas[n_layers - 2 - kk][j];
			}
		}

		delete[] activation_prime;
		activation_prime = new double[layers[n_layers - 2 - kk]];
		CUDA_1i1o(dsigmoid, activation_prime, fires[n_layers - 2 - kk], layers[n_layers - 2-kk]);

		delta = new double[layers[n_layers - 2 - kk]];
		CUDA_2i1o(mul, delta, activation_prime, error, layers[n_layers - 2 - kk]);

		deltas[n_layers - 3 - kk] = delta;
	}

	// Apply deltas
	for (int i = n_layers - 2; i > -1; i--)
		for (unsigned int j = 0; j < layers[i]; ++j)
			for (unsigned int k = 0; k < layers[i + 1]; k++)
				w[i][j][k] -= learning_rate * deltas[i][k] * fires[i][j];
}

uchar** read_mnist_images(string full_path, int& number_of_images, int& image_size){
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0, n_rows = 0, n_cols = 0;

		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2051) throw runtime_error("Invalid MNIST image file!");

		file.read((char *)&number_of_images, sizeof(number_of_images)), number_of_images = reverseInt(number_of_images);
		file.read((char *)&n_rows, sizeof(n_rows)), n_rows = reverseInt(n_rows);
		file.read((char *)&n_cols, sizeof(n_cols)), n_cols = reverseInt(n_cols);

		image_size = n_rows * n_cols;

		uchar** _dataset = new uchar*[number_of_images];
		for (int i = 0; i < number_of_images; i++) {
			_dataset[i] = new uchar[image_size];
			file.read((char *)_dataset[i], image_size);
		}
		return _dataset;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}

uchar* read_mnist_labels(string full_path, int& number_of_labels){
	auto reverseInt = [](int i) {
		unsigned char c1, c2, c3, c4;
		c1 = i & 255, c2 = (i >> 8) & 255, c3 = (i >> 16) & 255, c4 = (i >> 24) & 255;
		return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
	};

	typedef unsigned char uchar;

	ifstream file(full_path, ios::binary);

	if (file.is_open()) {
		int magic_number = 0;
		file.read((char *)&magic_number, sizeof(magic_number));
		magic_number = reverseInt(magic_number);

		if (magic_number != 2049) throw runtime_error("Invalid MNIST label file!");

		file.read((char *)&number_of_labels, sizeof(number_of_labels)), number_of_labels = reverseInt(number_of_labels);

		uchar* _dataset = new uchar[number_of_labels];
		for (int i = 0; i < number_of_labels; i++) {
			file.read((char*)&_dataset[i], 1);
		}
		return _dataset;
	}
	else {
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
}


int main(){
	cudaError_t cudaStatus;

	//// Setup Network
	const unsigned int n_layers = 3;
	const unsigned int layers[n_layers] = {784, 32, 10};

	const double learning_rate = .0001;

	double ***w = new double**[n_layers - 1];

	for (unsigned int i = 0; i < n_layers - 1; i++)
	{
		w[i] = new double*[layers[i]];

		for (unsigned int j = 0; j < layers[i]; ++j) {
			w[i][j] = new double[layers[i + 1]];

			for (unsigned int k = 0; k < layers[i + 1]; k++) {
				w[i][j][k] = ((rand() % 100) / 100.) / 5. - .1;

			}
		}
	}

	//// Train
	int N_TRAIN = 60000;
	int N_TEST = 10000;
	int IMG_SIZE = 784;

	string TRAIN_LABEL_PATH = "C:\\MNIST\\train-labels.idx1-ubyte";
	string TRAIN_IMAGE_PATH = "C:\\MNIST\\train-images.idx3-ubyte";
	string TEST_LABEL_PATH = "C:\\MNIST\\t10k-labels.idx1-ubyte";
	string TEST_IMAGE_PATH = "C:\\MNIST\\t10k-images.idx3-ubyte";

	uchar *temp_train_labels = read_mnist_labels(TRAIN_LABEL_PATH, N_TRAIN);
	uchar **temp_train_images = read_mnist_images(TRAIN_IMAGE_PATH, N_TRAIN, IMG_SIZE);

	double *train_labels = new double[N_TRAIN];
	double **train_images = new double*[N_TRAIN];
	for (int i = 0; i < N_TRAIN; i++) {
		train_labels[i] = (double)(temp_train_labels[i]);

		train_images[i] = new double[IMG_SIZE];
		for (int j = 0; j < IMG_SIZE; j++) {
			train_images[i][j] = (double)(temp_train_images[i][j]) / 255.;
		}
	}

	int N_CLASS = 10;

	unsigned int N_EP = 1;
	for (unsigned int e = 0; e < N_EP; e++) {
		double total_error = 0; 
		clock_t begin = clock();

		for (unsigned int i = 0; i < N_TRAIN; i++) {
			double **fires = forward(train_images[i], w, layers, n_layers);

			double *target = onehot(train_labels[i], N_CLASS);
			backward(w, target, fires, layers, n_layers, learning_rate);

			total_error += cross_entropy_loss(fires[n_layers-1], train_labels[i], N_CLASS);
		}
		clock_t end = clock();
		double secs = double(end - begin) / CLOCKS_PER_SEC;;
		
		printf("%i(%3.0fs): %f\n", e, secs, total_error);
	}
	
	//// Evaluate
	uchar *temp_test_labels = read_mnist_labels(TEST_LABEL_PATH, N_TEST);
	uchar **temp_test_images = read_mnist_images(TEST_IMAGE_PATH, N_TEST, IMG_SIZE);

	double *test_labels = new double[N_TEST];
	double **test_images = new double*[N_TEST];
	for (int i = 0; i < N_TEST; i++) {
		test_labels[i] = (double)(temp_test_labels[i]);
		
		test_images[i] = new double[IMG_SIZE];
		for (int j = 0; j < IMG_SIZE; j++)
			test_images[i][j] = (double)(temp_test_images[i][j]) / 255.;
	}

	int total = 0;
	int n_right = 0;
	for (unsigned int i = 0; i < N_TEST; i++) {
		double **fires = forward(test_images[i], w, layers, n_layers);

		double real = (double) amax(fires[n_layers - 1], N_CLASS);
		double target = test_labels[i];

		total += 1;
		if (real == target) {
			n_right += 1;
		}
	}
	printf("Correct: %f", ((float)n_right) / ((float)total));

	//// Cleanup
	// Exit profiling and tracing tools 
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}


template <typename T>
cudaError_t CUDA_1i1o(void(*func)(T*, const T*), T *b, const T *a, unsigned int size) {
	/*
	CUDA function wrapper.

	Parameters
	----------
	b: T*
		Output.
	a: T*
		Input.
	size: uint
		Number of values in a and b.
	*/
	T *dev_a = 0;
	T *dev_b = 0;
	cudaError_t cudaStatus;

	// Select GPU
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}


	// Allocate GPU Buffers
	// out
	cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// in
	cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(T));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Copy inputs from Host -> GPU Buffer
	cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(T), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}


	// Launch kernel w/ one thread per element.
	func << <1, size >> > (dev_b, dev_a);

	// Get kernel launch errors
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// Wait for kernel finish
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy outputs from GPU Buffer -> Host
	cudaStatus = cudaMemcpy(b, dev_b, size * sizeof(T), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_b);
	cudaFree(dev_a);

	return cudaStatus;
}


template <typename T>
cudaError_t CUDA_2i1o(void(*func)(T*, const T*, const T*), T *c, const T *a, const T *b, unsigned int size){
	/*
	CUDA function wrapper.

	Parameters
	----------
	c: T*
		Output.
	a: T*
		Input.
	b: T*
		Input.
	size: uint
		Number of values in a, b and c.
	*/
	T *dev_a = 0;
    T *dev_b = 0;
    T *dev_c = 0;
    cudaError_t cudaStatus;

    // Select GPU
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }


	// Allocate GPU Buffers
	// out
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// in
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

	// in
    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(T));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy inputs from Host -> GPU Buffer
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(T), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch kernel w/ one thread per element.
	func<<<1, size>>>(dev_c, dev_a, dev_b);

    // Get kernel launch errors
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

	// Wait for kernel finish
	cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

	// Copy outputs from GPU Buffer -> Host
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(T), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
