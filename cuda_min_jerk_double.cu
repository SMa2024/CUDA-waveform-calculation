#include <iostream>
#include <cuda.h>
#include <math.h>
# define PI           3.14159265358979323846

using namespace std ;

# define DELLEXPORT extern "C" __declspec(dllexport)

__global__ void cudaMinJerkKernel(float *intRampFn, float *rampFn, float * amp,float * Damp, float * freq,  float * Dfreq,
  float * phase, float * Dphase, float * sum, unsigned n_vectors, unsigned arr_size){
  
  unsigned idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx < arr_size){
    float temp = 0;
    double sampleRate = 400;
    double dt = 1/sampleRate;
    for (unsigned i = 0; i < n_vectors; i++)
      temp += (Damp[i]*rampFn[idx]+amp[i])*sinf((double)(Dfreq[i]*intRampFn[idx] + freq[i]*(idx+1)*dt +
       Dphase[i]*rampFn[idx] + phase[i]));

    sum[idx] = temp;}
}

  
  

DELLEXPORT void cudaInitialize(){
  
  cudaFree(0);

}


DELLEXPORT void cudaRearrange(float *intRampFn, float *rampFn, float * amp, float * Damp, float * freq,  float * Dfreq,
  float * phase, float * Dphase, float * output, int arr_size, int n_vector){

  const int nTPB = 1024;

  const long long int ARRAY_BYTES = arr_size * sizeof(float) ;
  const long long int Param_BYTES = n_vector * sizeof(float) ;

  float *amp_d, *freq_d, *phase_d, *d_out ,*d_intRamp, *d_Ramp,*Damp_d, *Dfreq_d, *Dphase_d;

  cudaMalloc((void **) &amp_d, Param_BYTES) ;
  cudaMalloc((void **) &freq_d, Param_BYTES) ;
  cudaMalloc((void **) &phase_d, Param_BYTES) ;
  cudaMalloc((void **) &Damp_d, Param_BYTES) ;
  cudaMalloc((void **) &Dfreq_d, Param_BYTES) ;
  cudaMalloc((void **) &Dphase_d, Param_BYTES) ;

  cudaMalloc((void **) &d_out, ARRAY_BYTES) ;
  cudaMalloc((void **) &d_intRamp, ARRAY_BYTES) ;
  cudaMalloc((void **) &d_Ramp, ARRAY_BYTES) ;

  cudaMemcpy(amp_d, amp, Param_BYTES, cudaMemcpyHostToDevice) ;
  cudaMemcpy(freq_d, freq, Param_BYTES, cudaMemcpyHostToDevice) ;
  cudaMemcpy(phase_d, phase, Param_BYTES, cudaMemcpyHostToDevice) ;
  cudaMemcpy(Damp_d, Damp, Param_BYTES, cudaMemcpyHostToDevice) ;
  cudaMemcpy(Dfreq_d, Dfreq, Param_BYTES, cudaMemcpyHostToDevice) ;
  cudaMemcpy(Dphase_d, Dphase, Param_BYTES, cudaMemcpyHostToDevice) ;

  cudaMemcpy(d_intRamp, intRampFn, ARRAY_BYTES, cudaMemcpyHostToDevice) ;
  cudaMemcpy(d_Ramp, rampFn, ARRAY_BYTES, cudaMemcpyHostToDevice) ;

  cudaMinJerkKernel<<<(arr_size + nTPB -1 )/nTPB,nTPB>>>(d_intRamp, d_Ramp, amp_d, Damp_d, freq_d, Dfreq_d, phase_d, Dphase_d,
    d_out,n_vector,arr_size) ;

  cudaMemcpy(output, d_out, ARRAY_BYTES, cudaMemcpyDeviceToHost) ;

  cudaFree(amp_d);
  cudaFree(freq_d);
  cudaFree(phase_d);
  cudaFree(Damp_d);
  cudaFree(Dfreq_d);
  cudaFree(Dphase_d);

  cudaFree(d_Ramp);
  cudaFree(d_intRamp);
  cudaFree(d_out);
}



