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
#define ARR_LEN			  15000
#define SPARSE        3500
#define TOL						1e-6
#define TILE_WIDTH    20
#define BLK_WIDTH     100
#define UERROR       1.11e-16

#define IMUL(a, b) __mul24(a, b)

void init_sym_matrix(double *arr, int len, int seed);
void lanczos_on_host(double *mat, double *eigs, int len);
void lanczos1_on_host(double *mat, int len);
void eigenvalues(double *alpha_vec, double* beta_vec, int num);


/*
__global__ void kernel (double* Md, double* Nd, double *Pd, int Width) {
	int Row = blockIdx.y*TILE_WIDTH + threadIdx.y;
  int Col = blockIdx.x*TILE_WIDTH + threadIdx.x;

  double Pvalue = 0;
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
	double elapsed_gpu, elapsed_gpu1;
	
	// Arrays on GPU global memory
	double *d_mat;
  double *d_alpha;
  double *d_beta;

	// Arrays on the host memory
	double *h_mat;
  double *h_alpha;
  double *h_beta;
  double *h_eigs1;
  double *h_eigs2;

  double result;
	
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
	size_t allocSize = totalLen * sizeof(double);
/*
	CUDA_SAFE_CALL(cudaMalloc((void **)&d_mat, allocSize));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_alpha, arrLen));
  CUDA_SAFE_CALL(cudaMalloc((void **)&d_beta, arrLen));
*/		
	// Allocate arrays on host memory
	h_mat = (double *) malloc(allocSize);
  //h_alpha = (double *) malloc(arrLen);
  //h_beta = (double *) malloc(arrLen);
  h_eigs1 = (double *) malloc(arrLen/100);
  //h_eigs2 = (double *) malloc(arrLen/100);
	
	// Initialize the host arrays
	printf("\nInitializing the arrays ...");
	// Arrays are initialized with a known seed for reproducability
	init_sym_matrix(h_mat, arrLen, 1);
	printf("\t... done\n\n");

  /*
  for(i = 0; i<ARR_LEN; i++) {
    for (j=0; j<ARR_LEN; j++) {
		printf("%f ",h_mat[i*ARR_LEN + j]);
	  }
    printf("\n");
  }
  */
	
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
	CUDA_SAFE_CALL(cudaMemcpy(d_mat, h_mat, allocSize, cudaMemcpyHostToDevice));
  
  dim3 dimBlock(TILE_WIDTH,TILE_WIDTH,1);
  dim3 dimGrid(BLK_WIDTH,BLK_WIDTH);

  cudaEventRecord(start1, 0);
	  
	// Launch the kernel
	kernel<<<dimGrid, dimBlock>>>(d_mat, d_alpha, d_beta, ARR_LEN);

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
*/
  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time1);
  
  lanczos_on_host(h_mat, h_eigs1, ARR_LEN);

  clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time2);
  time_stamp = diff(time1,time2);	
  printf("\nCPU time: %ld (nsec)", (long int)((double)(GIG * time_stamp.tv_sec + time_stamp.tv_nsec)));
	
  /*
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
	//free(h_mat);

  //cudaDeviceReset();

		
	return 0;
}

void init_sym_matrix(double *arr, int len, int seed) {
	int i, j;
	double randNum;
	srand(seed);

  double *transpose = (double *) malloc(len * len * sizeof(double));

	for (i=0; i<len; i++) {
    for (j=0; j<len; j++) {
		  randNum = (j % SPARSE == 0) ? (double) (rand() % 100000000) : 0;
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

  for (i=1; i<=num; i++) {
    if (alpha_vec[i] - abs(beta_vec[i]) - abs(beta_vec[i-1]) < y1) y1 = alpha_vec[i] - abs(beta_vec[i]) - abs(beta_vec[i-1]);
    if (alpha_vec[i] + abs(beta_vec[i]) + abs(beta_vec[i-1]) > z1) z1 = alpha_vec[i] + abs(beta_vec[i]) + abs(beta_vec[i-1]);
  }

  for (i=1; i<=num; i++) {
    double x;
    double y = y1;
    //printf("\ny = %f", y);
    double z = z1;
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

void lanczos_on_host(double *mat, double *eigs, int len) {
  double *w_vec = (double *) calloc(len, sizeof(double));
  double *v_vec = (double *) calloc(len, sizeof(double));
  double *alpha_vec = (double *) malloc(len * sizeof(double));
  double *beta_vec = (double *) malloc(len * sizeof(double));
  beta_vec[0] = 1;

  int k = 0;
  int i, j;
  double tmp;

  for (i=0; i<len; i++) {
    w_vec[i] = 1/sqrt(ARR_LEN);
  }

  while (abs(beta_vec[k]) > 100 || k==0 ) {
    printf("\nit start = %i", k);
    if  (k != 0) {
      for (i=0; i<=len; i++) {
        tmp = w_vec[i];
        w_vec[i] = v_vec[i]/beta_vec[k];
        v_vec[i] = -1 * beta_vec[k] * tmp;
      }
    }
    //v = v + A.mult(w)
    for (i=0; i<len; i++) {
		  for (j=0; j<len; j++) {
			  v_vec[i] += mat[i*len + j] * w_vec[j];
		  }
	  }
    k++;
    //alpha_vec[k] = (w transpose times v)

    alpha_vec[k] = 0;
    for (i=0; i<len; i++) {
		   alpha_vec[k] += w_vec[i] * v_vec[i];
	  }
    
    //v = v - alpha_vec[k]*w

    for (i=0; i<len; i++) {
		  v_vec[i] -= alpha_vec[k] * w_vec[i];
	  }

    // beta_vec[k] = norm of v_vec

    double sum = 0;
	  for (i=0; i<len; i++) {
		  sum += v_vec[i] * v_vec[i];
	  }
	  beta_vec[k] = sqrt(sum);
  }

  printf("\n final k = %i", k);
  
  beta_vec[k] = 0;
  beta_vec[0] = 0; //for eigs compute 

  /*
  printf("\n betavec = ");
  for (j=0; j<len; j++) {
    printf("%f, ", beta_vec[j]);
  }
  printf("\n alphavec = ");
  for (j=0; j<len; j++) {
    printf("%f, ", alpha_vec[j]);
  }
  */

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
