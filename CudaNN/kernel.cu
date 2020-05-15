/*
Deep neural network accelerated with CUDA.

C++ & CUDA
*/

#define _CRTDBG_MAP_ALLOC
#include <stdlib.h>
#include <crtdbg.h>

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


double** read_mnist_images(string full_path, int& number_of_images, int& image_size);
double* read_mnist_labels(string full_path, int& number_of_labels);

int amax(const double *real, const unsigned int &n_values);
double* onehot(const double &target, const int &N_CLASS);
__global__ void mul(double *output, const double *in1, const double *in2);
__global__ void matmul_n(double *output, const double *x, const double *mat, const unsigned int maty, const unsigned int matx);
__global__ void matmul_m(double *output, const double *x, const double *mat, const unsigned int maty, const unsigned int matx);

double cross_entropy_loss(double *real, const double &target, const unsigned int &n_outputs);
__global__ void dcross_entropy_loss(double *output, const double *real, const double *target, const unsigned int n_outputs);

__global__ void sigmoid(double *output, const double *input);
__global__ void dsigmoid(double *output, const double *input);

__global__ void update_w(double *w, double *fires, double *deltas, const double learning_rate, const unsigned int maty, const unsigned int matx) {
	for (unsigned int y = 0; y < maty; y++)
		for (unsigned int x = 0; x < matx; x++)
			w[y * matx + x] -= learning_rate * fires[y] * deltas[x];
}

class NN {
  private:
	const double learning_rate;
	const unsigned int n_layers;
    const unsigned int *layers;

	double **gpu_w;
	double **gpu_recent_fires;

  public:
	NN(const double lr, const unsigned int n_l, const unsigned int *l)
		: learning_rate(lr), n_layers(n_l), layers(l){
		
		// Setup W
		double **w = new double*[n_layers - 1];
		
		for (unsigned int i = 0; i < n_layers - 1; i++)
		{
			w[i] = new double[layers[i] * layers[i + 1]];

			for (unsigned int y = 0; y < layers[i]; y++) {
				for (unsigned int x = 0; x < layers[i + 1]; x++) {
					w[i][y * layers[i + 1] + x] = ((rand() % 100) / 100.) / 5. - .1;
				}
			}
		}
		
		// Copy W to GPU
		gpu_w = new double*[n_layers - 1];
		for (int i = 0; i < n_layers - 1; i++) {
			int size = layers[i] * layers[i + 1];

			cudaMalloc((void**)&gpu_w[i], size * sizeof(double));
			cudaMemcpy(gpu_w[i], w[i], size * sizeof(double), cudaMemcpyHostToDevice);

			delete[] w[i];
		}
		delete[] w;

		// Initialize GPU Buffers
		gpu_recent_fires = new double*[n_layers];
		for (int i = 0; i < n_layers; i++)
			cudaMalloc((void**)&gpu_recent_fires[i], layers[i] * sizeof(double));

	}

	~NN() {
		// Free GPU Buffers
		for (int i = 0; i < n_layers; i++)
			cudaFree(gpu_recent_fires[i]);
		delete[] gpu_recent_fires;

	}

	double *forward(const double *x) {
		/*
		Forward propogate input through network.

		Parameters
		----------
		x: double[layers[0]]
			Input vector.

		Returns
		-------
		double* Firing magnitude of output neurons.
		*/
		//
		cudaSetDevice(0);

		//
		cudaMemcpy(gpu_recent_fires[0], x, layers[0] * sizeof(double), cudaMemcpyHostToDevice);

		//
		for (unsigned int i = 0; i < n_layers - 1; i++) {
			matmul_n << <1, layers[i + 1] >> > (gpu_recent_fires[i + 1], gpu_recent_fires[i], gpu_w[i], layers[i], layers[i + 1]);
			cudaDeviceSynchronize();

			sigmoid << <1, layers[i + 1] >> > (gpu_recent_fires[i + 1], gpu_recent_fires[i + 1]);
			cudaDeviceSynchronize();
		}

		//
		double *output = new double[layers[n_layers - 1]];
		cudaMemcpy(output, gpu_recent_fires[n_layers-1], layers[n_layers-1] * sizeof(double), cudaMemcpyDeviceToHost);

		return output;
	}

	void backward(const double *target) {
		/*
		Calculate and backward propogate error.

		Parameters
		----------
		target: double[N_CLASS]
			Expected label.

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

		double *gpu_activation_prime = 0;
		cudaMalloc((void**)&gpu_activation_prime, layers[n_layers - 1] * sizeof(double));

		double *gpu_error = 0;
		cudaMalloc((void**)&gpu_error, layers[n_layers - 1] * sizeof(double));

		double **gpu_deltas = new double*[n_layers - 1];
		for (int i = 0; i < n_layers - 1; i++)
			cudaMalloc((void**)&gpu_deltas[i], (int)layers[i + 1] * sizeof(double));

		//// Output layer
		dcross_entropy_loss << <1, layers[n_layers - 1] >> > (gpu_error, gpu_recent_fires[n_layers - 1], gpu_target, layers[n_layers - 1]);
		cudaDeviceSynchronize();

		dsigmoid << <1, layers[n_layers - 1] >> > (gpu_activation_prime, gpu_recent_fires[n_layers - 1]);
		cudaDeviceSynchronize();

		mul << <1, layers[n_layers - 1] >> > (gpu_deltas[n_layers - 2], gpu_activation_prime, gpu_error);
		cudaDeviceSynchronize();

		//// Backpropogate
		for (int k = n_layers - 3; k > -1; k--) {
			//
			cudaFree(gpu_error);
			cudaMalloc((void**)&gpu_error, layers[k + 1] * sizeof(double));
			cudaFree(gpu_activation_prime);
			cudaMalloc((void**)&gpu_activation_prime, layers[k + 1] * sizeof(double));

			//
			matmul_m << <1, layers[k + 1] >> > (gpu_error, gpu_deltas[k + 1], gpu_w[k + 1], layers[k + 1], layers[k + 2]);
			cudaDeviceSynchronize();

			dsigmoid << <1, layers[k + 1] >> > (gpu_activation_prime, gpu_recent_fires[k + 1]);
			cudaDeviceSynchronize();

			mul << <1, layers[k + 1] >> > (gpu_deltas[k], gpu_activation_prime, gpu_error);
			cudaDeviceSynchronize();
		}

		// Apply deltas
		for (int i = 0; i < n_layers - 1; i++){
			update_w << <1, 1 >> > (gpu_w[i], gpu_recent_fires[i], gpu_deltas[i], learning_rate, layers[i], layers[i+1]);
			cudaDeviceSynchronize();
		}
		
		cudaError_t cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "w launch failed: %s\n", cudaGetErrorString(cudaStatus));
		}

		// Cleanup
		cudaFree(gpu_target);
		cudaFree(gpu_activation_prime);
		cudaFree(gpu_error);

		for (int i = 0; i < n_layers - 1; i++) {
			cudaFree(gpu_deltas[i]);
		}
		delete[] gpu_deltas;
	}
};


int main(){
	string TRAIN_LABEL_PATH = "C:\\MNIST\\train-labels.idx1-ubyte";
	string TRAIN_IMAGE_PATH = "C:\\MNIST\\train-images.idx3-ubyte";
	string TEST_LABEL_PATH = "C:\\MNIST\\t10k-labels.idx1-ubyte";
	string TEST_IMAGE_PATH = "C:\\MNIST\\t10k-images.idx3-ubyte";

	unsigned int N_EP = 1;

	const double learning_rate = .0001;
	const unsigned int n_layers = 3;
	const unsigned int layers[] = { 784, 32, 10 };

	NN nn(learning_rate, n_layers, layers);

	// Train
	int IMG_SIZE, N_TRAIN;;
	unsigned int N_CLASS = layers[n_layers - 1];

	double *train_labels = read_mnist_labels(TRAIN_LABEL_PATH, N_TRAIN);
	double **train_images = read_mnist_images(TRAIN_IMAGE_PATH, N_TRAIN, IMG_SIZE);

	for (unsigned int e = 0; e < N_EP; e++) {
		double total_error = 0; 
		clock_t begin = clock();

		for (unsigned int i = 0; i < N_TRAIN; i++) {
			double *real = nn.forward(train_images[i]);

			double *target = onehot(train_labels[i], N_CLASS);
			nn.backward(target);

			total_error += cross_entropy_loss(real, train_labels[i], N_CLASS);
		}
		double secs = double(clock() - begin) / CLOCKS_PER_SEC;
		
		printf("%i(%3.0fs): %f\n", e, secs, total_error);
	}
	
	// Evaluate
	int N_TEST;

	double *test_labels = read_mnist_labels(TEST_LABEL_PATH, N_TEST);
	double **test_images = read_mnist_images(TEST_IMAGE_PATH, N_TEST, IMG_SIZE);

	int n_correct = 0, real;
	for (unsigned int i = 0; i < N_TEST; i++) {
		real = amax(nn.forward(test_images[i]), N_CLASS);

		if (real == test_labels[i])
			n_correct += 1;
	}
	printf("Correct: %f", n_correct / ((float) N_TEST));

	// Cleanup
	cudaDeviceReset();

	_CrtDumpMemoryLeaks();

    return 0;
}


double** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
	/*
	Read MNIST dataset images.

	Parameters
	----------
	full_path: string
		Path to MNIST dataset.

	Effects
	-------
	Set number of images to number of images in dataset, image size to dataset image size.
	*/
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

double* read_mnist_labels(string full_path, int& number_of_labels) {
	/*
	Read MNIST dataset images.

	Parameters
	----------
	full_path: string
		Path to MNIST dataset.

	Effects
	-------
	Set number of labels to number of labels in dataset, image size to dataset image size.
	*/
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
			train_labels[i] = (double)_dataset[i];

		return train_labels;
	}
	else {
		throw runtime_error("Unable to open file `" + full_path + "`!");
	}
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
	double max_value = -9999;

	for (unsigned int i = 0; i < n_values; i++) {
		if (real[i] > max_value) {
			max_value = real[i];
			max_idx = i;
		}
	}

	return max_idx;
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
	double* Onehot encoding of target.
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

__global__ void mul(double *output, const double *in1, const double *in2) {
	/*
	Vector elementwise multiplication.

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

__global__ void matmul_n(double *output, const double *x, const double *mat, const unsigned int maty, const unsigned int matx) {
	/*
	Matrix multiplication.

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
		output[i] += x[j] * mat[j * matx + i];
	}
}

__global__ void matmul_m(double *output, const double *x, const double *mat, const unsigned int maty, const unsigned int matx) {
	/*
	Matrix multiplication

	Parameters
	----------
	x: double[m]
	mat: double[m, n]
	maty: int = m
	matx: int = n

	Returns
	-------
	double[n] Output of matrix multiplication.
	*/
	int i = threadIdx.x;

	output[i] = 0;

	for (unsigned int j = 0; j < matx; j++) {
		output[i] += x[j] * mat[i * matx + j];
	}
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
	double Cross entropy loss.
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
	double[n] Derivative of cross entropy loss.
	*/
	int i = threadIdx.x;

	output[i] = (real[i] - target[i]) / n_outputs;
}

__global__ void sigmoid(double *output, const double *input) {
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

__global__ void dsigmoid(double *output, const double *input) {
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
