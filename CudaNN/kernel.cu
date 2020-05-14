/*
Deep neural network accelerated with CUDA.

C++ & CUDA
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


double** read_mnist_images(string full_path, int& number_of_images, int& image_size) {
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
	double max = -9999;

	for (unsigned int i = 0; i < n_values; i++) {
		if (real[i] > max) {
			max = real[i];
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

__global__ void matmul_n(double *output, const double *x, const double *mat, const unsigned int maty, const unsigned int matx) {
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
	double[m] Output of matrix multiplication.
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

class NN {
  private:
	const double learning_rate;
	const unsigned int n_layers;
    const unsigned int *layers;

	double **w;
	double **gpu_recent_fires;

  public:
	NN(const double lr, const unsigned int n_l, const unsigned int *l)
		: learning_rate(lr), n_layers(n_l), layers(l){
		
		w = new double*[n_layers - 1];
		
		for (unsigned int i = 0; i < n_layers - 1; i++)
		{
			w[i] = new double[layers[i] * layers[i + 1]];

			for (unsigned int y = 0; y < layers[i]; y++) {
				for (unsigned int x = 0; x < layers[i + 1]; x++) {
					w[i][y * layers[i + 1] + x] = ((rand() % 100) / 100.) / 5. - .1;
				}
			}
		}
	}

	double *forward(double *x) {
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
		for (int i = 0; i < n_layers - 1; i++) {
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
			matmul_n << <1, layers[i + 1] >> > (gpu_fires[i + 1], gpu_fires[i], gpu_w[i], layers[i], layers[i + 1]);
			cudaDeviceSynchronize();

			sigmoid << <1, layers[i + 1] >> > (gpu_fires[i + 1], gpu_fires[i + 1]);
			cudaDeviceSynchronize();
		}

		//
		double *output = new double[layers[n_layers - 1]];
		cudaMemcpy(output, gpu_fires[n_layers-1], layers[n_layers-1] * sizeof(double), cudaMemcpyDeviceToHost);

		for (int i = 0; i < n_layers - 1; i++)
			cudaFree(gpu_w[i]);
		delete[] gpu_w;

		gpu_recent_fires = gpu_fires;

		return output;
	}

	void backward(const double *target) {
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
		double **gpu_w = new double*[n_layers - 1];
		for (int i = 0; i < n_layers - 1; i++) {
			int size = layers[i] * layers[i + 1];

			cudaMalloc((void**)&gpu_w[i], size * sizeof(double));
			cudaMemcpy(gpu_w[i], w[i], size * sizeof(double), cudaMemcpyHostToDevice);
		}

		double *gpu_target = 0;
		cudaMalloc((void**)&gpu_target, layers[n_layers - 1] * sizeof(double));
		cudaMemcpy(gpu_target, target, layers[n_layers - 1] * sizeof(double), cudaMemcpyHostToDevice);

		double *gpu_error = 0;
		cudaMalloc((void**)&gpu_error, layers[n_layers - 1] * sizeof(double));
		double *gpu_activation_prime = 0;
		cudaMalloc((void**)&gpu_activation_prime, layers[n_layers - 1] * sizeof(double));

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
			cudaFree(gpu_error);
			cudaMalloc((void**)&gpu_error, layers[k + 1] * sizeof(double));
			cudaFree(gpu_activation_prime);
			cudaMalloc((void**)&gpu_activation_prime, layers[k + 1] * sizeof(double));

			matmul_m << <1, layers[k + 1] >> > (gpu_error, gpu_deltas[k + 1], gpu_w[k + 1], layers[k + 1], layers[k + 2]);
			cudaDeviceSynchronize();

			dsigmoid << <1, layers[k + 1] >> > (gpu_activation_prime, gpu_recent_fires[k + 1]);
			cudaDeviceSynchronize();

			mul << <1, layers[k + 1] >> > (gpu_deltas[k], gpu_activation_prime, gpu_error);
			cudaDeviceSynchronize();
		}
		double **deltas = new double*[n_layers - 1];
		for (int i = 0; i < n_layers - 1; i++) {
			deltas[i] = new double[layers[i + 1]];
			cudaMemcpy(deltas[i], gpu_deltas[i], layers[i + 1] * sizeof(double), cudaMemcpyDeviceToHost);
			cudaFree(gpu_deltas[i]);
		}
		delete[] gpu_deltas;

		//
		double **fires = new double*[n_layers];
		for (int i = 0; i < n_layers; i++) {
			fires[i] = new double[layers[i]];
			cudaMemcpy(fires[i], gpu_recent_fires[i], layers[i] * sizeof(double), cudaMemcpyDeviceToHost);
		}

		// Apply deltas
		for (int i = 0; i < n_layers - 1; i++)
			for (unsigned int y = 0; y < layers[i]; y++)
				for (unsigned int x = 0; x < layers[i + 1]; x++)
					w[i][y * layers[i + 1] + x] -= learning_rate * fires[i][y] * deltas[i][x];

		//
		for (int i = 0; i < n_layers - 1; i++)
			cudaFree(gpu_w[i]);
		delete[] gpu_w;

		for (int i = 0; i < n_layers - 1; i++) {
			delete[] deltas[i];
			cudaFree(gpu_recent_fires[i]);
		}
		delete[] gpu_recent_fires;
		delete[] deltas;

		cudaFree(gpu_target);
		cudaFree(gpu_activation_prime);
		cudaFree(gpu_error);
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

    return 0;
}
