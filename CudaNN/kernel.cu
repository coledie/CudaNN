/*
Deep neural network accelerated with CUDA.

C++11 & CUDA
*/
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <ctime>
#include <string>
#include <fstream>


using namespace std;

typedef unsigned char uchar;

/*
TODO
- Backprop fully gpu based
* error backprop, weight update
- Object Oriented
- C++ & CUDA memory leaks
*/

template <typename T>
cudaError_t CUDA_1i1o(void(*func)(T*, const T*), T *&b, const T *a, unsigned int size);
template <typename T>
cudaError_t CUDA_2i1o(void (*func)(T*, const T*, const T*), T *&c, const T *a, const T *b, unsigned int size);

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

__global__ void matmul(double *output, const double *x, const double *mat, const unsigned int maty, const unsigned int matx) {
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
	int i = threadIdx.x;

	output[i] = 0;

	for (unsigned int j = 0; j < maty; j++) {
		output[i] += x[j] * mat[(j*maty) + i];
	}
}

double *matmul_old(const double *x, double *mat, const unsigned int &maty, const unsigned int &matx) {
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
		output[i] = 0;

		for (unsigned int j = 0; j < maty; j++) {
			output[i] += x[j] * mat[j * matx + i];
		}
	}

	return output;
}

double cross_entropy_loss(double *real, const double &target, const unsigned int &n_outputs) {
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

	for (unsigned int i = 0; i < n_outputs; i++)
		total += exp(real[i]);

	return -real[(int)target] + log(total);
}

__global__ void dcross_entropy_loss(double *output, const double *real, const double *target, const unsigned int n_outputs) {
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
	int i = threadIdx.x;

	output[i] = (real[i] - target[i]) / n_outputs;
}

double* onehot(const double &target, const int &N_CLASS) {
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

int amax(const double *real, const unsigned int &n_values) {
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

double **forward(double *x, double **w, const unsigned int *layers, const unsigned int &n_layers) {
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
	// Select GPU
	cudaSetDevice(0);

	// Allocate GPU Buffers
	double **gpu_w = new double*[n_layers - 1];
	for (int i = 0; i < n_layers-1; i++) {
		int size = layers[i] * layers[i + 1];

		cudaMalloc((void**)&gpu_w[i], size * sizeof(double));
		cudaMemcpy(gpu_w[i], w[i], size * sizeof(double), cudaMemcpyHostToDevice);
	}

	double **gpu_fires = new double*[n_layers];
	for (int i = 0; i < n_layers; i++)
		cudaMalloc((void**)&gpu_fires[i], layers[i] * sizeof(double));
	cudaMemcpy(gpu_fires[0], x, layers[0] * sizeof(double), cudaMemcpyHostToDevice);
	
	//
	for (unsigned int i = 0; i < n_layers - 1; i++) {
		
		double *old_fires = new double[layers[i]];
		cudaMemcpy(old_fires, gpu_fires[i], layers[i] * sizeof(double), cudaMemcpyDeviceToHost);
		double *temp = new double[layers[i+1]];
		temp = matmul_old(old_fires, w[i], layers[i], layers[i+1]);
		cudaMemcpy(gpu_fires[i+1], temp, layers[i+1] * sizeof(double), cudaMemcpyHostToDevice);
		/*
		matmul<<<1, layers[i + 1] >>>(gpu_fires[i+1], gpu_fires[i], gpu_w[i], layers[i], layers[i+1]);		
		cudaDeviceSynchronize();
		*/
		sigmoid<<<1, layers[i + 1] >>>(gpu_fires[i + 1], gpu_fires[i + 1]);
		cudaDeviceSynchronize();
	}

	//
	double **fires = new double*[n_layers];
	for (int i = 0; i < n_layers; i++) {
		fires[i] = new double[layers[i]];
		cudaMemcpy(fires[i], gpu_fires[i], layers[i] * sizeof(double), cudaMemcpyDeviceToHost);
		cudaFree(gpu_fires[i]);
	}
	delete[] gpu_fires;

	for (int i = 0; i < n_layers - 1; i++)
		cudaFree(gpu_w[i]);
	delete[] gpu_w;

	return fires;
}

void backward(double **w, const double *target, double **fires, const unsigned int *layers, const unsigned int n_layers, const double learning_rate) {
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
	// Select GPU
	cudaSetDevice(0);

	// Allocate GPU Buffers
	double *gpu_target = 0;
	cudaMalloc((void**)&gpu_target, layers[n_layers - 1] * sizeof(double));
	cudaMemcpy(gpu_target, target, layers[n_layers - 1] * sizeof(double), cudaMemcpyHostToDevice);

	double *gpu_error = 0;
	cudaMalloc((void**)&gpu_error, layers[n_layers - 1] * sizeof(double));
	double *gpu_activation_prime = 0;
	cudaMalloc((void**)&gpu_activation_prime, layers[n_layers - 1] * sizeof(double));
	double *gpu_delta = 0;
	cudaMalloc((void**)&gpu_delta, (int)layers[n_layers - 1] * sizeof(double));

	double **gpu_fires = new double*[n_layers];
	for (int i = 0; i < n_layers; i++) {
		cudaMalloc((void**)&gpu_fires[i], layers[i] * sizeof(double));
		cudaMemcpy(gpu_fires[i], fires[i], layers[i] * sizeof(double), cudaMemcpyHostToDevice);
	}

	//
	double **deltas = new double*[n_layers - 1];

	//// Output layer
	dcross_entropy_loss << <1, layers[n_layers - 1] >> > (gpu_error, gpu_fires[n_layers - 1], gpu_target, layers[n_layers - 1]);
	cudaDeviceSynchronize();

	dsigmoid << <1, layers[n_layers - 1] >> > (gpu_activation_prime, gpu_fires[n_layers - 1]);
	cudaDeviceSynchronize();

	mul << <1, layers[n_layers - 1] >> > (gpu_delta, gpu_activation_prime, gpu_error);
	cudaDeviceSynchronize();

	deltas[n_layers - 2] = new double[layers[n_layers - 1]];
	cudaMemcpy(deltas[n_layers - 2], gpu_delta, layers[n_layers - 1] * sizeof(double), cudaMemcpyDeviceToHost);

	//// Backpropogate
	for (int k = n_layers - 3; k > -1; k--) {
		int kk = n_layers - 3 - k;

		double *error;
		//error = matmul(deltas[n_layers - 2 - kk], w[k + 1], layers[n_layers - 1 - kk], layers[n_layers - 2 - kk]);
		error = new double[layers[n_layers - 2 - kk]];
		for (int i = 0; i < layers[n_layers - 2 - kk]; i++)
		{
			error[i] = 0;

			for (int j = 0; j < layers[n_layers - 1 - kk]; j++) {

				error[i] += w[k + 1][i * layers[n_layers - 1 - kk] + j] * deltas[n_layers - 2 - kk][j];
			}
		}

		cudaFree(gpu_activation_prime);
		cudaMalloc((void**)&gpu_activation_prime, layers[n_layers - 2 - kk] * sizeof(double));

		cudaFree(gpu_delta);
		cudaMalloc((void**)&gpu_delta, layers[n_layers - 2 - kk] * sizeof(double));

		cudaFree(gpu_error);
		cudaMalloc((void**)&gpu_error, layers[n_layers - 2 - kk] * sizeof(double));
		cudaMemcpy(gpu_error, error, layers[n_layers - 2 - kk] * sizeof(double), cudaMemcpyHostToDevice);
		delete[] error;

		dsigmoid << <1, layers[n_layers - 2 - kk] >> > (gpu_activation_prime, gpu_fires[n_layers - 2 - kk]);
		cudaDeviceSynchronize();

		mul << <1, layers[n_layers - 2 - kk] >> > (gpu_delta, gpu_activation_prime, gpu_error);
		cudaDeviceSynchronize();

		deltas[n_layers - 3 - kk] = new double[layers[n_layers - 2 - kk]];
		cudaMemcpy(deltas[n_layers - 3 - kk], gpu_delta, layers[n_layers - 2 - kk] * sizeof(double), cudaMemcpyDeviceToHost);
	}

	// Apply deltas
	for (int i = n_layers - 2; i > -1; i--)
		for (unsigned int y = 0; y < layers[i]; y++)
			for (unsigned int x = 0; x < layers[i + 1]; x++)
				w[i][y * layers[i+1] + x] -= learning_rate * fires[i][y] * deltas[i][x];

	//
	for (int i = 0; i < n_layers - 1; i++) {
		delete[] deltas[i];
		cudaFree(gpu_fires[i]);
	}
	delete[] gpu_fires;
	delete[] deltas;

	cudaFree(gpu_target);
	cudaFree(gpu_activation_prime);
	cudaFree(gpu_delta);
	cudaFree(gpu_error);
}

double** read_mnist_images(string full_path, int& number_of_images, int& image_size){
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

		//
		double **train_images = new double*[number_of_images];
		for (int i = 0; i < number_of_images; i++) {
		
			train_images[i] = new double[image_size];

			for (int j = 0; j < image_size; j++) {
				train_images[i][j] = (double)(_dataset[i][j]) / 255.;
			}
		}

		return train_images;
	}
	else {
		throw runtime_error("Cannot open file `" + full_path + "`!");
	}
}

double* read_mnist_labels(string full_path, int& number_of_labels){
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

		//
		double *train_labels = new double[number_of_labels];
		for (int i = 0; i < number_of_labels; i++)
			train_labels[i] = (double) _dataset[i];

		return train_labels;
	}
	else {
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
}


int main(){
	//// Setup
	const double learning_rate = .0001;

	const unsigned int n_layers = 3;
	const unsigned int layers[n_layers] = {784, 32, 10};

	double **w = new double*[n_layers - 1];

	for (unsigned int i = 0; i < n_layers - 1; i++)
	{
		w[i] = new double[layers[i] * layers[i+1]];

		for (unsigned int y = 0; y < layers[i]; y++) {
			for (unsigned int x = 0; x < layers[i + 1]; x++) {
				w[i][y * layers[i+1] + x] = ((rand() % 100) / 100.) / 5. - .1;
			}
		}
	}

	//// Train
	int IMG_SIZE = 784;
	int N_TRAIN = 60000;
	unsigned int N_EP = 1;

	string TRAIN_LABEL_PATH = "C:\\MNIST\\train-labels.idx1-ubyte";
	string TRAIN_IMAGE_PATH = "C:\\MNIST\\train-images.idx3-ubyte";

	double *train_labels = read_mnist_labels(TRAIN_LABEL_PATH, N_TRAIN);
	double **train_images = read_mnist_images(TRAIN_IMAGE_PATH, N_TRAIN, IMG_SIZE);

	unsigned int N_CLASS = layers[n_layers - 1];

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
	int N_TEST = 10000;

	string TEST_LABEL_PATH = "C:\\MNIST\\t10k-labels.idx1-ubyte";
	string TEST_IMAGE_PATH = "C:\\MNIST\\t10k-images.idx3-ubyte";

	double *test_labels = read_mnist_labels(TEST_LABEL_PATH, N_TEST);
	double **test_images = read_mnist_images(TEST_IMAGE_PATH, N_TEST, IMG_SIZE);

	int n_right = 0; int real; double **fires;
	for (unsigned int i = 0; i < N_TEST; i++) {
		fires = forward(test_images[i], w, layers, n_layers);

		real = amax(fires[n_layers - 1], N_CLASS);

		if (real == test_labels[i])
			n_right += 1;
	}
	printf("Correct: %f", ((float)n_right) / ((float)N_TEST));

	//// Cleanup
	cudaDeviceReset();

    return 0;
}


template <typename T>
cudaError_t CUDA_1i1o(void(*func)(T*, const T*), T *&b, const T *a, unsigned int size) {
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
	cudaSetDevice(0);
	
	// Allocate GPU Buffers
	cudaMalloc((void**)&dev_b, size * sizeof(T));
	cudaMalloc((void**)&dev_a, size * sizeof(T));
	
	// Copy inputs from Host -> GPU Buffer
	cudaMemcpy(dev_a, a, size * sizeof(T), cudaMemcpyHostToDevice);
	
	// Launch kernel w/ one thread per element.
	func << <1, size >> > (dev_b, dev_a);

	// Get kernel launch errors
	cudaGetLastError();
	
	// Wait for kernel finish
	cudaDeviceSynchronize();
	
	// Copy outputs from GPU Buffer -> Host
	b = new T[size];
	cudaMemcpy(b, dev_b, size * sizeof(T), cudaMemcpyDeviceToHost);

	cudaFree(dev_b);
	cudaFree(dev_a);

	return cudaStatus;
}

template <typename T>
cudaError_t CUDA_2i1o(void(*func)(T*, const T*, const T*), T *&c, const T *a, const T *b, unsigned int size){
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
    cudaSetDevice(0);
    
	// Allocate GPU Buffers
	cudaMalloc((void**)&dev_c, size * sizeof(T));

	cudaMalloc((void**)&dev_a, size * sizeof(T));
    cudaMalloc((void**)&dev_b, size * sizeof(T));

    // Copy inputs from Host -> GPU Buffer
    cudaMemcpy(dev_a, a, size * sizeof(T), cudaMemcpyHostToDevice);

    cudaMemcpy(dev_b, b, size * sizeof(T), cudaMemcpyHostToDevice);

    // Launch kernel w/ one thread per element.
	func<<<1, size>>>(dev_c, dev_a, dev_b);

    // Get kernel launch errors
    cudaGetLastError();

	// Wait for kernel finish
	cudaDeviceSynchronize();

	// Copy outputs from GPU Buffer -> Host
	c = new T[size];
    cudaMemcpy(c, dev_c, size * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
