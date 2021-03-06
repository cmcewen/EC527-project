//nvcc -Xcompiler -fopenmp -arch=sm_20 -o eigenvalue eigenvalue.cu

#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <omp.h>
#include <driver_functions.h>

#define GIG 1000000000
#define CPG 2.533327           // Cycles per GHz -- Adjust to your computer

// Assertion to check for errors
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"CUDA_SAFE_CALL: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}


#define NUM_THREADS_PER_BLOCK 	256
#define NUM_BLOCKS 				1
#define PRINT_TIME 				1
#define ARR_LEN			  16384
#define RESULT_LEN    ARR_LEN/100
#define SPARSE        10240
#define TOL						1e-6
#define TILE_WIDTH    20
#define BLK_WIDTH     100
#define UERROR        1.11e-16
#define NUM_THREADS   4

#define IMUL(a, b) __mul24(a, b)

void init_sym_matrix(double *arr, int len, int seed);
void lanczos(double *mat, double *eigs, int len);
void lanczos_optimized(double *mat, double *eigs, int len);
void lanczos_omp(double *mat, double *eigs, int len);
void lanczos_GPU(double *mat, double *eigs, int len);
void eigenvalues(double *alpha_vec, double* beta_vec, double* eigs, int num);

__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void first_op (double *w_vec, double *v_vec, double* beta_vec, int *d_k) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;  
  double tmp;
  int k = *d_k;
  tmp = w_vec[index];
  w_vec[index] = v_vec[index]/beta_vec[k];
  v_vec[index] = -1 * beta_vec[k] * tmp;
} 

__global__ void mat_mult (double* A, double *w_vec, double *v_vec) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;  
  double tmp = 0;
  for (int j=0; j<ARR_LEN; j++) {
    tmp += A[index*ARR_LEN + j] * w_vec[j];
  }
  v_vec[index] += tmp;
}

__global__ void inc_k(int *d_k) {
  printf("\n inc k");
  (*d_k)++;
}

__global__ void wtv (double *w_vec, double *v_vec, double *d_result) {
  __shared__ double dot_temp[NUM_THREADS_PER_BLOCK];
  int index = threadIdx.x + blockIdx.x * blockDim.x;  
  dot_temp[threadIdx.x] = w_vec[index] * v_vec[index];

  __syncthreads();
  if (threadIdx.x == 0) {
    double sum = 0;
    for (int i = 0; i < NUM_THREADS_PER_BLOCK; i++)
      sum += dot_temp[i];
    atomicAdd(d_result, sum);
  }
}

__global__ void assign_alpha(double *alpha_vec, double *d_result, int *d_k) {
  int k = *d_k;
  alpha_vec[k] = *d_result;
  printf("assign alpha = %f", *d_result);
  *d_result = 0;
}

__global__ void vsub (double *w_vec, double *v_vec, double *alpha_vec, int *d_k) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;  
  int k = *d_k;
  v_vec[index] -= alpha_vec[k] * w_vec[index];
}

__global__ void norm (double *v_vec, double *d_result) {
  __shared__ double dot_temp[NUM_THREADS_PER_BLOCK];
  int index = threadIdx.x + blockIdx.x * blockDim.x;  
  dot_temp[threadIdx.x] = v_vec[index] * v_vec[index];

  __syncthreads();
  if (threadIdx.x == 0) {
    double sum = 0;
    for (int i = 0; i < NUM_THREADS_PER_BLOCK; i++)
      sum += dot_temp[i];
    atomicAdd(d_result, sum);
  }
}

__global__ void assign_beta(double *beta_vec, double *d_result, int *d_k) {
  int k = *d_k;
  double r = *d_result;
  beta_vec[k] = sqrt(r);
  *d_result = 0;
}

int main(int argc, char **argv){

  struct timespec diff(struct timespec start, struct timespec end);
  struct timespec time1, time2;
  struct timespec time_stamp;
		
	// GPU Timing variables
	cudaEvent_t start, stop, start1, stop1;
	float elapsed_gpu, elapsed_gpu1;
	
	// Arrays on GPU global memory
	double *d_mat;
  double *d_alpha;
  double *d_beta;
  double *d_w_vec;
  double *d_v_vec;
  int *d_k;
  double *d_result;

	// Arrays on the host memory
	double *h_mat;
  double *h_alpha;
  double *h_beta;
  double *h_eigs1;
  double *h_w_vec;


	printf("Length of the array = %d\n", ARR_LEN);

	// Allocate GPU memory
	size_t matAllocSize = ARR_LEN * ARR_LEN * sizeof(double);
  size_t vecAllocSize = ARR_LEN * sizeof(double);
  size_t svecAllocSize = RESULT_LEN * sizeof(double);
/*
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mat, matAllocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_alpha, svecAllocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_beta, svecAllocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_w_vec, vecAllocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_v_vec, vecAllocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_k, sizeof(int)));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_result, sizeof(double)));
*/		
	// Allocate arrays on host memory
	h_mat = (double *) malloc(matAllocSize);
//  h_alpha = (double *) malloc(svecAllocSize);
 // h_beta = (double *) malloc(svecAllocSize);
 // h_w_vec = (double *) malloc(vecAllocSize);
  h_eigs1 = (double *) malloc(svecAllocSize);
	
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducability
	init_sym_matrix(h_mat, ARR_LEN, 1);
	printf("\t... done\n\n");
	
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
	
  for (int i=0; i<ARR_LEN; i++) {
    h_w_vec[i] = 1/sqrt(ARR_LEN);
  }
  h_beta[0] = 1;
  double beta_test = 500;
  int h_k = 0;

	// Transfer the arrays to the GPU memory
	CUDA_SAFE_CALL(cudaMemcpy(d_mat, h_mat, matAllocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_w_vec, h_w_vec, vecAllocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_beta, h_beta, svecAllocSize, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_k, &h_k, sizeof(int), cudaMemcpyHostToDevice));
  cudaEventRecord(start1, 0);
	  
	// Launch the kernels
  while ((abs(beta_test) > 100) && (h_k < 50)) {
    printf("\n it = %i", h_k);
	  if (h_k != 0) {
      first_op<<<ARR_LEN/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(d_w_vec, d_v_vec, d_beta, d_k);
      cudaDeviceSynchronize();
      CUDA_SAFE_CALL(cudaPeekAtLastError());
    }
    mat_mult<<<ARR_LEN/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(d_mat, d_w_vec, d_v_vec);
    cudaDeviceSynchronize();
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
      inc_k<<<1, 1>>>(d_k);
      cudaDeviceSynchronize();
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
      h_k++;
      wtv<<<ARR_LEN/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(d_w_vec, d_v_vec, d_result);
      cudaDeviceSynchronize();
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
      assign_alpha<<<1, 1>>>(d_alpha, d_result, d_k);
      cudaDeviceSynchronize();
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
      vsub<<<ARR_LEN/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(d_w_vec, d_v_vec, d_alpha, d_k);
      cudaDeviceSynchronize(); 
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
      cudaDeviceSynchronize();    
      norm<<<ARR_LEN/NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK>>>(d_v_vec, d_result);
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
      cudaDeviceSynchronize();
      assign_beta<<<1, 1>>>(d_beta, d_result, d_k);
          cudaDeviceSynchronize();
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
      cudaDeviceSynchronize();
      cudaMemcpy(&beta_test, &(d_beta[h_k]), sizeof(double), cudaMemcpyDeviceToHost);
      CUDA_SAFE_CALL(cudaMemcpy(h_beta, d_beta, svecAllocSize, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
	  CUDA_SAFE_CALL(cudaPeekAtLastError());
    printf("\n beta = %f", beta_test);
        printf("\n betaarr = %f", h_beta[h_k]);
    printf("\n h_k = %i", h_k);
      cudaDeviceSynchronize();
	  CUDA_SAFE_CALL(cudaPeekAtLastError());

  }   

	// Check for errors during launch
	CUDA_SAFE_CALL(cudaPeekAtLastError());

  cudaEventRecord(stop1, 0);
	
	// Transfer the results back to the host
	CUDA_SAFE_CALL(cudaMemcpy(h_alpha, d_alpha, svecAllocSize, cudaMemcpyDeviceToHost));
  CUDA_SAFE_CALL(cudaMemcpy(h_beta, d_beta, svecAllocSize, cudaMemcpyDeviceToHost));

  eigenvalues(h_alpha, h_beta, h_eigs1, h_k);
  printf("\n");
	
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
*/	

	// Compute the results on the host
  memset(h_eigs1, 0, sizeof(h_eigs1));
  
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
  lanczos(h_mat, h_eigs1, ARR_LEN);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  time_stamp = diff(time1,time2);	
  printf("\nRegular time: %ld (nsec)\n", (long int)((double)(GIG * time_stamp.tv_sec + time_stamp.tv_nsec)));

  memset(h_eigs1, 0, sizeof(h_eigs1));

	init_sym_matrix(h_mat, ARR_LEN, 1);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
  lanczos_optimized(h_mat, h_eigs1, ARR_LEN);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  time_stamp = diff(time1,time2);	
  printf("\nOptimized time: %ld (nsec)\n", (long int)((double)(GIG * time_stamp.tv_sec + time_stamp.tv_nsec)));

  memset(h_eigs1, 0, sizeof(h_eigs1));
	
  init_sym_matrix(h_mat, ARR_LEN, 1);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
  lanczos_omp(h_mat, h_eigs1, ARR_LEN);
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  time_stamp = diff(time1,time2);	
  printf("\nOMP time: %ld (nsec)\n", (long int)((double)(GIG * time_stamp.tv_sec + time_stamp.tv_nsec)));


	// Free-up device and host memory
/*
	CUDA_SAFE_CALL(cudaFree(d_mat));
  CUDA_SAFE_CALL(cudaFree(d_alpha));
  CUDA_SAFE_CALL(cudaFree(d_beta));
  CUDA_SAFE_CALL(cudaFree(d_w_vec));
  CUDA_SAFE_CALL(cudaFree(d_v_vec));
  CUDA_SAFE_CALL(cudaFree(d_k));
  CUDA_SAFE_CALL(cudaFree(d_result));

	free(h_mat);
	free(h_alpha);
	free(h_beta);
	free(h_eigs1);
*/
  cudaDeviceReset();

	return 0;
}

void init_sym_matrix(double *arr, int len, int seed) {
	int i, j;
	double randNum;
	srand(seed);

  double *transpose = (double *) malloc(len * len * sizeof(double));

	for (i=0; i<len; i++) {
    for (j=0; j<len; j++) {
		  randNum = (j % SPARSE == 0) ? (double) (rand() % 100000) : 0;
      // randNum = (j % SPARSE == 0) ? (double) rand() : 0;
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

void eigenvalues(double *alpha_vec, double* beta_vec, double* eigs, int num) {
  double *p_vals = (double *) malloc((num+1) * sizeof(double));
  p_vals[0] = 1;

  int j, i, changes;

  double y1 = 0;
  double z1 = 0;
  double x, y, z;

  for (i=1; i<=num; i++) {
    if (alpha_vec[i] - abs(beta_vec[i]) - abs(beta_vec[i-1]) < y1) y1 = alpha_vec[i] - abs(beta_vec[i]) - abs(beta_vec[i-1]);
    if (alpha_vec[i] + abs(beta_vec[i]) + abs(beta_vec[i-1]) > z1) z1 = alpha_vec[i] + abs(beta_vec[i]) + abs(beta_vec[i-1]);
  }

  for (i=1; i<=num; i++) {
    y = y1;
    //printf("\ny = %f", y);
    (i ==1) ? z = z1 : z = eigs[i-1]+1;
    //printf("\nz = %f", z);
    while( abs(z-y) > UERROR*(abs(y) + abs(z))) {
      x = (y+z)/2;
      //printf("\nx = %f", x);
      //calculate a
      p_vals[1] = alpha_vec[1] - x;
      for (j=2; j<=num; j++) {
        p_vals[j] = (alpha_vec[j]-x)*p_vals[j-1]-beta_vec[j-1]*beta_vec[j-1]*p_vals[j-2];      
      }
      changes = 0;
      //printf("\npvals = ");
      for (j=1; j<=num; j++) {
        //printf("%0.1f ", p_vals[j]);
        if ((p_vals[j] > 0) && (p_vals[j-1] > 0));
        else if ((p_vals[j] > 0) && (p_vals[j-1] < 0)) changes++;
        else if ((p_vals[j] < 0) && (p_vals[j-1] < 0));
        else if ((p_vals[j] < 0) && (p_vals[j-1] > 0)) changes++;
        else if ((p_vals[j] == 0)) changes++;
      }
      //printf("\n i = %i, changes = %i", i, changes);
      if (changes > num - i) z = x;
      else y=x;
    }
    eigs[i] = x;
    printf("\n eig %i = %f", i, x);
  } 
  printf("\n");
}

void lanczos(double *mat, double *eigs, int len) {
  double *w_vec = (double *) calloc(len, sizeof(double));
  double *v_vec = (double *) calloc(len, sizeof(double));
  double *alpha_vec = (double *) malloc(len * sizeof(double));
  double *beta_vec = (double *) malloc(len * sizeof(double));
  beta_vec[0] = 1;

  int k = 0;
  int i, j;
  double tmp, b, tmp0;

  for (i=0; i<len; i++) {
    w_vec[i] = 1/sqrt(ARR_LEN);
  }

  while (abs(beta_vec[k]) > 100 || k==0 ) {
    //printf("\nit start = %i", k);
    if  (k != 0) {
      b = beta_vec[k];
      for (i=0; i<=len; i++) {
        tmp = w_vec[i];
        w_vec[i] = v_vec[i]/b;
        v_vec[i] = -1 * b * tmp;
      }
    }
    //v = v + A.mult(w)
    for (i=0; i<len; i++) {
      tmp0 = 0;
		  for (j=0; j<len; j++) {
			  tmp0 += mat[i*len + j] * w_vec[j];
		  }
      v_vec[i] += tmp0;
	  }
    k++;
    //alpha_vec[k] = (w transpose times v)

    tmp0 = 0;
    for (i=0; i<len; i++) {
		   tmp0 += w_vec[i] * v_vec[i]; 
	  }
    alpha_vec[k] = tmp0;    

    //v = v - alpha_vec[k]*w

    tmp0 = 0;
    for (i=0; i<len; i++) {
		  v_vec[i] -= alpha_vec[k] * w_vec[i];
	  }

    // beta_vec[k] = norm of v_vec

    tmp0 = 0;
	  for (i=0; i<len; i++) {
		  tmp0 += v_vec[i] * v_vec[i]; 
	  }
	  beta_vec[k] = sqrt(tmp0);
  }

  printf("\n final k = %i", k);
  
  beta_vec[k] = 0;
  beta_vec[0] = 0; //for eigs compute 

  printf("\n%0.30f, %0.30f", alpha_vec[1], beta_vec[1]);
  for (j=0; j<k-2; j++) printf(", 0");
  printf("\n");
  for (j=0; j<k-1; j++) {
    for (i=0; i<j; i++) {
      printf("0, ");
    }
    if (j == k-2) printf("%0.30f, %0.30f", beta_vec[j+1], alpha_vec[j+2]);
    else printf("%0.30f, %0.30f, %0.30f", beta_vec[j+1], alpha_vec[j+2], beta_vec[j+2]);
    for (i=j+3; i<k; i++) {
      printf(", 0");
    }
    printf("\n");
  }

  eigenvalues(alpha_vec, beta_vec, eigs, k);
  printf("\n");
}

void lanczos_optimized(double *mat, double *eigs, int len) {
  double *w_vec = (double *) calloc(len, sizeof(double));
  double *v_vec = (double *) calloc(len, sizeof(double));
  double *alpha_vec = (double *) malloc(len * sizeof(double));
  double *beta_vec = (double *) malloc(len * sizeof(double));
  beta_vec[0] = 1;

  int k = 0;
  int i, j;
  double tmp, b, tmp0, tmp1, tmp2, tmp3;

  for (i=0; i<len; i++) {
    w_vec[i] = 1/sqrt(ARR_LEN);
  }

  while (abs(beta_vec[k]) > 500 || k==0 ) {
    //printf("\nit start = %i", k);
    if  (k != 0) {
      b = beta_vec[k];
      for (i=0; i<=len; i++) {
        tmp = w_vec[i];
        w_vec[i] = v_vec[i]/b;
        v_vec[i] = -1 * b * tmp;
      }
    }
    //v = v + A.mult(w)
    for (i=0; i<len; i++) {
      tmp0 = tmp1 = tmp2 = tmp3 = 0;
		  for (j=0; j<len; j+=4) {
			  tmp0 += mat[i*len + j] * w_vec[j];
			  tmp1 += mat[i*len + j+1] * w_vec[j+1];
			  tmp2 += mat[i*len + j+2] * w_vec[j+2];
			  tmp3 += mat[i*len + j+3] * w_vec[j+3];
		  }
      v_vec[i] += tmp0 + tmp1 + tmp2 + tmp3;
	  }
    k++;
    //alpha_vec[k] = (w transpose times v)

    tmp0 = tmp1 = tmp2 = tmp3 = 0;
    for (i=0; i<len; i+=4) {
		   tmp0 += w_vec[i] * v_vec[i]; 
		   tmp1 += w_vec[i+1] * v_vec[i+1];
		   tmp2 += w_vec[i+2] * v_vec[i+2];
		   tmp3 += w_vec[i+3] * v_vec[i+3];
	  }
    alpha_vec[k] = tmp0 + tmp1 + tmp2 + tmp3;    

    //v = v - alpha_vec[k]*w

    for (i=0; i<len; i+=4) {
		  v_vec[i] -= alpha_vec[k] * w_vec[i];
		  v_vec[i+1] -= alpha_vec[k] * w_vec[i+1];
		  v_vec[i+2] -= alpha_vec[k] * w_vec[i+2];
		  v_vec[i+3] -= alpha_vec[k] * w_vec[i+3];
	  }

    // beta_vec[k] = norm of v_vec

    tmp0 = tmp1 = tmp2 = tmp3 = 0;
	  for (i=0; i<len; i+=4) {
		  tmp0 += v_vec[i] * v_vec[i]; 
		  tmp1 += v_vec[i+1] * v_vec[i+1];
		  tmp2 += v_vec[i+2] * v_vec[i+2];
		  tmp3 += v_vec[i+3] * v_vec[i+3];
	  }
	  beta_vec[k] = sqrt(tmp0 + tmp1 + tmp2 + tmp3);
  }

  printf("\n final k = %i", k);
  
  beta_vec[k] = 0;
  beta_vec[0] = 0; //for eigs compute 

  eigenvalues(alpha_vec, beta_vec, eigs, k);
  printf("\n");
}

void lanczos_omp(double *mat, double *eigs, int len) {
  double *w_vec = (double *) calloc(len, sizeof(double));
  double *v_vec = (double *) calloc(len, sizeof(double));
  double *alpha_vec = (double *) malloc(len * sizeof(double));
  double *beta_vec = (double *) malloc(len * sizeof(double));
  beta_vec[0] = 1;

  int k = 0;
  double tmp, b, tmp0;

  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel for
  for (int i=0; i<len; i++) {
    w_vec[i] = 1/sqrt(ARR_LEN);
  }

  while (abs(beta_vec[k]) > 500 || k==0 ) {
    //printf("\nit start = %i", k);
    if  (k != 0) {
      b = beta_vec[k];
      #pragma omp parallel for
      for (int i=0; i<=len; i++) {
        tmp = w_vec[i];
        w_vec[i] = v_vec[i]/b;
        v_vec[i] = -1 * b * tmp;
      }
    }
    //v = v + A.mult(w)
    for (int i=0; i<len; i++) {
      double tmp1 = 0;
		  for (int j=0; j<len; j++) {
			  tmp1 += mat[i*len + j] * w_vec[j];
		  }
      v_vec[i] += tmp1;
	  }
    k++;
    //alpha_vec[k] = (w transpose times v)

    tmp0 = 0;
    for (int i=0; i<len; i++) {
		   tmp0 += w_vec[i] * v_vec[i]; 
	  }
    alpha_vec[k] = tmp0;    

    //v = v - alpha_vec[k]*w

    #pragma omp parallel for
    for (int i=0; i<len; i++) {
		  v_vec[i] -= alpha_vec[k] * w_vec[i];
	  }

    // beta_vec[k] = norm of v_vec

    tmp0 = 0;
	  for (int i=0; i<len; i++) {
		  tmp0 += v_vec[i] * v_vec[i]; 
	  }
	  beta_vec[k] = sqrt(tmp0);
  }

  printf("\n final k = %i", k);
  
  beta_vec[k] = 0;
  beta_vec[0] = 0; //for eigs compute 

  eigenvalues(alpha_vec, beta_vec, eigs, k);
  printf("\n");
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
