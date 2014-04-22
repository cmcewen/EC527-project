//nvcc -o cuda_MMM cuda_MMM.cu

#include <cstdio>
#include <cstdlib>
#include <math.h>

#define GIG 1000000000
#define CPG 2.533327           // Cycles per GHz -- Adjust to your computer

// Assertion to check for errors

/*
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}
*/

#define NUM_THREADS_PER_BLOCK 	256
#define NUM_BLOCKS 				1
#define PRINT_TIME 				1
#define ARR_LEN			  10
#define TOL						1e-6
#define TILE_WIDTH    20
#define BLK_WIDTH     100

#define IMUL(a, b) __mul24(a, b)

void init_sym_matrix(float *arr, int len, int seed);
void lanczos_on_host(float *mat, int len);


/*
__global__ void kernel (float* Md, float* Nd, float *Pd, int Width) {
	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

  float Pvalue = 0;
  for (int k = 0; k < Width; ++k)
    Pvalue += Md[Row*Width+k] * Nd[k*Width+Col];

  Pd[Row*Width+Col] = Pvalue;
}
*/

int main(int argc, char **argv){

  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp;

	int arrLen = 0;
  int totalLen = 0;
		
	// GPU Timing variables
	cudaEvent_t start, stop, start1, stop1;
	float elapsed_gpu, elapsed_gpu1;
	
	// Arrays on GPU global memory
	float *d_m;
  float *d_n;
  float *d_p;

	// Arrays on the host memory
	float *h_mat;

  float result;
	
	int i, j, errCount = 0, zeroCount = 0;
	
	if (argc > 1) {
		arrLen  = atoi(argv[1]);
	}
	else {
		arrLen = ARR_LEN;
    totalLen = arrLen * arrLen;
	}

	printf("Length of the array = %d\n", arrLen);

	// Allocate GPU memory
	size_t allocSize = totalLen * sizeof(float);

	/*
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_m, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_n, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_p, allocSize));

  */
		
	// Allocate arrays on host memory
	h_mat = (float *) malloc(allocSize);
	
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducability
	init_sym_matrix(h_mat, arrLen, 1);
	printf("\t... done\n\n");

  for(i = 0; i<ARR_LEN; i++) {
    for (j=0; j<ARR_LEN; j++) {
		printf("%f, ",h_mat[i*ARR_LEN + j]);
	  }
    printf("\n");
  }

  lanczos1_on_host(h_mat, ARR_LEN);
	
	/*
#if PRINT_TIME
	// Create the cuda events
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
  cudaEventCreate(&start1);
	cudaEventCreate(&stop1);
	// Record event on the default stream
	cudaEventRecord(start, 0);
#endif
	
	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_m, h_m, allocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_n, h_n, allocSize, cudaMemcpyHostToDevice));
  
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
  dim3 dimGrid(BLK_WIDTH,BLK_WIDTH);

  cudaEventRecord(start1, 0);
	  
	// Launch the kernel
	kernel<<<dimGrid, dimBlock>>>(d_m, d_n, d_p, ARR_LEN);

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());

  cudaEventRecord(stop1, 0);
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_p_gpu, d_p, allocSize, cudaMemcpyDeviceToHost));
	
#if PRINT_TIME
	// Stop and destroy the timer
	cudaEventRecord(stop,0);
	cudaEventSynchronize(stop);
  cudaEventSynchronize(stop1);
	cudaEventElapsedTime(&elapsed_gpu, start, stop);
  cudaEventElapsedTime(&elapsed_gpu1, start1, stop1);
  printf("\nArray size: %i ", ARR_LEN);
	printf("\nGPU time: %f (msec)", elapsed_gpu);
  printf("\nGPU time (w/o transfer): %f (msec)", elapsed_gpu1);
	cudaEventDestroy(start);
  cudaEventDestroy(start1);
	cudaEventDestroy(stop);
  cudaEventDestroy(stop1);
#endif
	
	// Compute the results on the host

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
  
  MatrixMulOnHost(h_m, h_n, h_p_test, ARR_LEN);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  time_stamp = diff(time1,time2);	
  printf("\nCPU time: %ld (nsec)", (long int)((double)(GIG * time_stamp.tv_sec + time_stamp.tv_nsec)));

  float largestDif = 0;	
  
	// Compare the results
	for(i = 0; i < totalLen; i++) {
		if (abs(h_p_gpu[i] - h_p_test[i]) > .001 * h_p_test[i]) {
			errCount++;
		}
    if ( abs(h_p_gpu[i] - h_p_test[i]) / h_p_test[i] > largestDif) {
      largestDif = abs(h_p_gpu[i] - h_p_test[i]) / h_p_test[i];
    }
	}
	
	
	for(i = 0; i < 50; i++) {
		printf("%d:\t%.8f\t%.8f\n", i, h_p_test[i], h_p_gpu[i]);
	}
	
	if (errCount > 0) {
		printf("\n@ERROR: TEST FAILED: %d results did not matched\n", errCount);
	}
	else if (zeroCount > 0){
		printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
	}
	else {
		printf("\nTEST PASSED: All results matched");
	}

  printf("\nLargest dif = %f percent\n", largestDif);
	
	// Free-up device and host memory
	CUDA_SAFE_CALL(cudaFree(d_m));
  CUDA_SAFE_CALL(cudaFree(d_n));
  CUDA_SAFE_CALL(cudaFree(d_p));
	*/
	free(h_mat);

  //cudaDeviceReset();

		
	return 0;
}

void init_sym_matrix(float *arr, int len, int seed) {
	int i, j;
	float randNum;
	srand(seed);

  float *transpose = (float *) malloc(len * len * sizeof(float));

	for (i=0; i<len; i++) {
    for (j=0; j<len; j++) {
		  randNum = (float) rand();
		  arr[i*len+j] = randNum;
      transpose[j*len+i] = arr[i*len+j];
    }
	}
  for (i=0; i<len; i++) {
    for (j=0; j<len; j++) {
      arr[i*len+j] += transpose[i*len+j];
    }
  }

  free(transpose);
}

void vadd_Aw (float *v, float *w, float *A, int len) {
	for (int i=0; i<len; i++) {
		for (int j=0; j<len; j++) {
			v[i] += A[i*len + j] * w[j];
		}
	}
}

float wtr_v (float *w, float* v, int len) {
	float sum = 0;
	for (int i=0; i<len; i++) {
		sum += w[i] * v[i];
	}
	return sum;
}

void v_sub_alphaw (float* v, float* w, float alphaK, int len) {
	for (int i=0; i<len; i++) {
		v[i] -= alphaK * w[i];
	}
}

float vec_norm(float *vec, int len) {
	float sum = 0;
	for (int i=0; i<len; i++) {
		sum += vec[i];
	}
	return sum;
}

void lanczos_on_host(float *mat, int len) {
  float *w_vec = (float *) calloc(len * sizeof(float));
  float *v_vec = (float *) calloc(len * sizeof(float));
  float *alpha_vec = (float *) malloc(len * sizeof(float));
  float *beta_vec = (float *) malloc(len * sizeof(float));
  w_vec[0] = 1;
  beta_vec[0] = 1;

  int k = 0;
  int i;
  float tmp;

  while (beta_vec[k] != 0) {
    if  (k != 0) {
      for (i=0; i<len; i++) {
        tmp = w_vec[i];
        w_vec[i] = v_vec[i]/beta_vec[k];
        v_vec[i] = -1 * beta_vec[k] * tmp;
      }
    }
    //v = v + A.mult(w)
    printf("\nmult.add ");
    vadd_Aw(v_vec, w_vec, mat, len);
    k++;
    //alpha_vec[k] = (w transpose times v)
    printf("\nwtr_v ");
    alpha_vec[k] = wtr_v(w_vec, v_vec, len);
    //v = v - alpha_vec[k]*w
    printf("\nvsub");
    v_sub_alphaw(v_vec, w_vec, alpha_vec[k], len);
    // beta_vec[k] = norm of v_vec
    printf("\nvec_norm");
    beta_vec[k] = vec_norm(v_vec, len);
  }

  printf("\nalpha_vec: ");
  for (i=0; i<len; i++) {
  	printf("%f, ", alpha_vec[i]);
  }
  printf("\nbeta_vec: ");
  for (i=0; i<len; i++) {
  	printf("%f, ", beta_vec[i]);
  }
}

float xt_x_mult(float *x, int len) {
	float sum = 0;
	for (int j=0; j<len; j++) {
		sum += x[j] * x[len + j];
	}
	return sum;
}

void vec_mat_mult(float *x, float *mat, int len) {
	for (int i=0; i<len; i++) {
		x[len+i] = 0;
		for (int j=0; j<len; j++) {
			x[len+i] += mat[i*len + j] * x[i];
		}
	}
}

void lanczos1_on_host(float *mat, int len) {
	float *x_vec = (float *) calloc(len * len * sizeof(float));
	float *beta_vec = (float *) calloc(len * sizeof(float));
	float *alpha_vec = (float *) calloc(len * sizeof(float));
	int i = 1;
	int j;
	beta_vec[0] = 1;
	beta_vec[1] = 1;

	while (beta_vec[i] != 0) {
		vec_mat_mult(x+len*(i-1), mat, len);
		alpha_vec[i] = xt_x_mult(x + len*(i-1), len);
		if (i > 1) {
			for (j=0; j<len; j++) {
				x[i*len + j] = x[i*len + j] - alpha_vec[i]*x[(i-1)*len + j];
			}
		}
		if (i > 2) {
			for (j=0; j<len; j++) {
				x[i*len + j] = x[i*len + j] - beta_vec[i-1]*x[(i-2)*len + j];
			}
		}
		beta_vec[i] = vec_norm(x_vec+len, len);
		//div_beta();
		for (j=0; j<len; j++) {
			x[i*len + j] = x[i*len + j] / beta_vec[i];
		}
		i++;
	}
	printf("\nalpha_vec: ");
  for (i=0; i<len; i++) {
  	printf("%f, ", alpha_vec[i]);
  }
  printf("\nbeta_vec: ");
  for (i=0; i<len; i++) {
  	printf("%f, ", beta_vec[i]);
  }

}

struct timespec diff(struct timespec start, struct timespec end)
{
  struct timespec temp;
  if ((end.tv_nsec-start.tv_nsec)<0) {
    temp.tv_sec = end.tv_sec-start.tv_sec-1;
    temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
  } else {
    temp.tv_sec = end.tv_sec-start.tv_sec;
    temp.tv_nsec = end.tv_nsec-start.tv_nsec;
  }
  return temp;
}
