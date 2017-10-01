/* 
 * This program performs a convolution of randomly generated data or sound data
 * from a .wav file if one is provided.  The convolution is performed on the
 * the CPU and then on the GPU. The speedup is calculated.  The program is
 * executed like this if no sound file is provided:
 * ./blur -t <thread_count> -b <block_count>
 * and like this if a sound file is provided:
 * ./blur -t <thread_count> -b <block_count> -i <sound file> -o <output sound file>
 *
 * The thread count specifies the number of threads in a block. The block count
 * specifies the maximum number of blocks.  
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <sndfile.h>
#include <cuda_runtime.h>

/* macro to check a cuda call and exit if the call generates an error */
#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

/* sound file struct */
typedef struct
{
   int sndFile;           //sound file given?
   char * outputFile;     //name of output file
   float * inputSndData;  //data from sound file
   float * outputSndData; //data to write to output file
   SF_INFO infInfo;       //info about input sound file
} sndFileT;

/* CPU functions */
float gaussian(float x, float mean, float std);
float * generateBlurVector();
int isDecDigits(char * str);
void parseCmdLineArgs(int argc, char **argv, int * blocks, int * threads,
                      char ** inFile, char ** outFile);
void verbose();
int checkArgs(int blocks, int threads, char * inFile, char * outFile);
void getSoundData(sndFileT * snd, char * inFile, char * outFile);
void getInputData(sndFileT * snd, float * h_inputData, int nFrames, 
                  int nChannels, int channel);
float cpuBlur(float * h_inputData, float * h_outputData, float * h_blurV, 
              int nFrames);
float gpuBlur(float * h_inputData, float * h_outputData, float * h_blurV, 
              int nFrames, int blocks, int threads);
void gaussianTests(sndFileT * snd, float * blurV, int blocks, int threads);
void compare(float * h_outputDataFromDev, float * h_outputData, int nFrames);
void writeSndData(sndFileT * snd);

/* GPU function */
__global__ void cudaBlurKernel(float * d_inputData, float * d_outputData, 
                               float * blurV, int nFrames);

#define PI 3.14159265358979f
#define GAUSSIAN_SIDE_WIDTH 10
#define GAUSSIAN_SIZE (2 * GAUSSIAN_SIDE_WIDTH + 1)


/* computes gaussian function */   
float gaussian(float x, float mean, float std) 
{
   return (1 / (std * sqrt(2 * PI))) *
          exp(-1.0 / 2.0 * pow((x - mean) / std, 2));
}

/* uses gaussian function to compute a blur vector */
float * generateBlurVector()
{
   //constants to use in gaussian function
   float mean = 0.0;
   float std = 5.0;
   int i;
   //create array to hold the blur vector values
   float * blurV = (float *) malloc(sizeof(float) * GAUSSIAN_SIZE);

   for (i = -GAUSSIAN_SIDE_WIDTH; i <= GAUSSIAN_SIDE_WIDTH; i++)
      blurV[ GAUSSIAN_SIDE_WIDTH + i ] = gaussian((float)i, mean, std);

   // Normalize to avoid clipping and/or hearing loss
   float total = 0.0;
   for (i = 0; i < GAUSSIAN_SIZE; i++)
      total += blurV[i];

   // Normalize by a factor of total
   for (i = 0; i < GAUSSIAN_SIZE; i++)
      blurV[i] /= total;

   return blurV;
}

/* returns 1 if the string contains the characters '0' - '9' */
int isDecDigits(char * str)
{
   int i;
   int len = strlen(str);
   for (i = 0; i < len; i++)
   {
      if (str[i] < '0' || str[i] > '9') return 0;
   }
   return 1;
}

/* parses the command line arguments to get the number of blocks, 
   threads, input file name, and output file name 
*/
void parseCmdLineArgs(int argc, char **argv, int * blocks, int * threads,
                      char ** inFile, char ** outFile)
{
   int i;
   for (i = 1; i < argc; i++)
   {
      if (strcmp(argv[i], "-b") == 0 && (i+1) < argc && isDecDigits(argv[i+1]))
         (*blocks) = atoi(argv[i+1]);
      else if (strcmp(argv[i], "-t") == 0 && (i+1) < argc && isDecDigits(argv[i+1]))
         (*threads) = atoi(argv[i+1]);
      else if (strcmp(argv[i], "-i") == 0 && (i+1) < argc)
         (*inFile) = argv[i+1];
      else if (strcmp(argv[i], "-o") == 0 && (i+1) < argc)
         (*outFile) = argv[i+1];
      else if (strcmp(argv[i], "-v") == 0)
         verbose();
   }
}

/* prints usage information and exits */
void verbose()
{
   printf("Usage: blur -b <number of blocks> -t <threads per block> ");
   printf("[-i <input wav file> -o <output wav file>]\n");
   exit(0);
}

/* checks command line arguments.
   Users must supply blocks and threads. The input file and output file
   are optional but must be supplied together if used.
*/
int checkArgs(int blocks, int threads, char * inFile, char * outFile)
{
   if (blocks == 0 || threads == 0) return 1;  
   if (inFile == NULL && outFile != 0) return 1;  
   if (inFile != NULL && outFile == 0) return 1;  
   return 0;
}

/* gets data for the convolution from either the data read from a sound file
   or randomly generates it.
*/
void getInputData(sndFileT *snd, float * inputData, 
                  int nFrames, int nChannels, int channel)
{
   int i;
   if (!(snd->sndFile))
   {
      for (i = 0; i < nFrames; i++)
         inputData[i] = ((float) rand()) / RAND_MAX;
   } else
   {
      for (i = 0; i < nFrames; i++)
         inputData[i] = snd->inputSndData[(i * nChannels) + channel];
   }
}

/* Opens the sound file and reads the contents, filling the
   sndFileT struct. The sound file may contain multiple channels,
   for example, stereo data.
*/
void getSoundData(sndFileT * snd, char * inFile, char * outFile)
{
   SNDFILE *inf; 
   SF_INFO infInfo; 
   int amtRead;

   // Open input audio file
   inf = sf_open(inFile, SFM_READ, &infInfo);
   if (!inf) 
   {
       printf("Cannot open input file: %s\n", inFile);
       verbose();
   }

   // Read audio
   snd->infInfo = infInfo;
   snd->outputFile = outFile;
   snd->inputSndData = (float *) malloc(sizeof(float) * infInfo.frames * infInfo.channels);
   snd->outputSndData = (float *) malloc(sizeof(float) * infInfo.frames * infInfo.channels);
   amtRead = sf_read_float(inf, snd->inputSndData, infInfo.frames * infInfo.channels);
   assert(amtRead == infInfo.frames * infInfo.channels);
   sf_close(inf);
}

/* performs the gaussian tests, first on the CPU and then on the GPU */
void gaussianTests(sndFileT * snd, float * h_blurV, int blocks, int threads)
{
   int i;
   float cpuTime, gpuTime;
   int nFrames = 1e7, nChannels = 1;  //defaults
   //if a sound file was given, use the frames and channels
   //in the sound file
   if (snd->sndFile) 
   {
      nChannels = snd->infInfo.channels; 
      nFrames = snd->infInfo.frames;
   }

   //Host side: per channel input data
   float * h_inputData = (float *) malloc(sizeof (float) * nFrames);
   float * h_outputData = (float *) malloc(sizeof (float) * nFrames);
   float * h_outputDataFromDev = (float *) malloc(sizeof(float) * nFrames);

   for (i = 0; i < nChannels; i++)
   {
      getInputData(snd, h_inputData, nFrames, nChannels, i);

      //perform the convolution on the CPU
      printf("CPU Blurring ....\n");
      cpuTime = cpuBlur(h_inputData, h_outputData, h_blurV, nFrames);

      //perform the convolution on the GPU
      printf("GPU Blurring ....\n");
      gpuTime = gpuBlur(h_inputData, h_outputDataFromDev, h_blurV, nFrames, 
                        blocks, threads);

      //compare the results to make sure they match
      printf("Comparing ... ");      
      compare(h_outputDataFromDev, h_outputData, nFrames);
      printf("outputs match.\n"); 
      printf("CPU time: %f milliseconds\n", cpuTime);
      printf("GPU time: %f milliseconds\n", gpuTime);
      printf("Speedup overall: %f\n",  cpuTime / gpuTime);

      //if a sound file was given, save the result to write
      //later to an output file
      if (snd->sndFile)
      {
         for (int j = 0; j < nFrames; j++)
            snd->outputSndData[j * nChannels + i] = h_outputDataFromDev[j];
      }
   }
   //if a sound file was given, save the result to the output file
   if (snd->sndFile)
   {
      writeSndData(snd);
   }
   //free the dynamically allocated data
   free(h_inputData);
   free(h_outputData);
   free(h_outputDataFromDev);
}

/* write the convoluted data to the output file */
void writeSndData(sndFileT * snd)
{
    SNDFILE *outFile;
    SF_INFO outInfo;
    int amt = snd->infInfo.frames * snd->infInfo.channels;
    outInfo = snd->infInfo;
    outFile = sf_open(snd->outputFile, SFM_WRITE, &outInfo);

    if (!outFile) 
    {
        printf("Cannot open output file, exiting\n");
        exit(EXIT_FAILURE);
    }
    sf_write_float(outFile, snd->outputSndData, amt);
    sf_close(outFile);
}

/* compare the output computed by the CPU to the output computed by the GPU 
   It should be almost exactly the same.
*/
void compare(float * h_outputDataFromDev, float * h_outputData, int nFrames)
{
   int i;
   for (i = 0; i < nFrames; i++) 
   {
      if (fabs(h_outputDataFromDev[i] - h_outputData[i]) >= 1e-6) 
      {
         printf("Incorrect output at index %d: host: %f, device: %f\n", 
                i, h_outputData[i], h_outputDataFromDev[i]);
         exit(0);
      }
   }
}

/* set up what is needed to perform the convolution on the GPU and launch
   the kernel to do the convolution.
*/
float gpuBlur(float * h_inputData, float * h_outputDataFromDev, float * h_blurV, int nFrames,
              int blocks, int threads)
{
   //To Do: Use the blocks and threads passed in to set the
   //       blocks and threads for the grid and the blocks in the grid.
   //       Make sure the number of threads in a block is not larger
   //       than the hardware allowable number of threads per block.
   //       The blocks provided by the user is the maximum number of
   //       blocks, but blocks * threads can not be larger the 
   //       number of nFrames. So blocks may need to be set to a
   //       a smaller value.
   threads = 0;  /* min(threads, .... ); */
   blocks =  0;  /* min(blocks, .....);  */

   //To Do: create the grid and the blocks (both one dimensional)

   cudaEvent_t start_gpu, stop_gpu;
   float gpuMsecTime = -1;

   float * d_inputData = NULL; 
   float * d_outputData = NULL;
   float * d_blurV = NULL; 

   //To Do: allocate the arrays for the input, output and blur vector on the
   //       device side and set the arrays to the appropriate values

   //use the cuda event functions for timing
   CHECK(cudaEventCreate(&start_gpu));
   CHECK(cudaEventCreate(&stop_gpu));
   CHECK(cudaEventRecord(start_gpu));

   //To Do: launch the kernel

   //check if the launch caused an error
   CHECK(cudaGetLastError());

   //wait until threads are finished and get the time
   CHECK(cudaEventRecord(stop_gpu));
   CHECK(cudaEventSynchronize(stop_gpu)); //wait until the GPU is done
   CHECK(cudaEventElapsedTime(&gpuMsecTime, start_gpu, stop_gpu));

   //To Do: copy the device output into the host h_outDataFromDev array
   CHECK(cudaMemcpy(h_outputDataFromDev, d_outputData, 
                    (nFrames * sizeof(float)), cudaMemcpyDeviceToHost));

   //To Do: free the dynamically allocate device arrays

   return gpuMsecTime;
}

/* perform the convolution on the gpu */
__global__ void cudaBlurKernel(float * d_inputData, float * d_outputData, 
                               float * d_blurV, int nFrames)
{
   /* To Do: Provide the code to perform the convolution on the device side. */
   /*        Use the cpuBlur function as a guide.                            */
}

/* perform the convolution on the cpu */
float cpuBlur(float * h_inputData, float * h_outputData, float * blurV, int nFrames)
{
   int i, j;
   float cpuMsecTime = -1;
   memset(h_outputData, 0, nFrames * sizeof (float));
   cudaEvent_t start_cpu, stop_cpu;
   CHECK(cudaEventCreate(&start_cpu));
   CHECK(cudaEventCreate(&stop_cpu));
   CHECK(cudaEventRecord(start_cpu));

   for (i = 0; i < GAUSSIAN_SIZE; i++) 
   {
      for (j = 0; j <= i; j++) 
      {
         h_outputData[i] += h_inputData[i - j] * blurV[j];
      }
   }
   for (i = GAUSSIAN_SIZE; i < nFrames; i++) 
   {
      for (j = 0; j < GAUSSIAN_SIZE; j++)
      {
         h_outputData[i] += h_inputData[i - j] * blurV[j];
      }
   }

   // Stop timer
   CHECK(cudaEventRecord(stop_cpu));
   CHECK(cudaEventSynchronize(stop_cpu));
   CHECK(cudaEventElapsedTime(&cpuMsecTime, start_cpu, stop_cpu));
   return cpuMsecTime;
}

/* to run the convolution, the main must be supplied, at the minimum,
   the number of blocks and threads created to perform the convolution
   on the gpu.  For example, 32 blocks and 1024 threads would be
   specified like this:
   ./blur -b 32 -t 1024
   Optionally, the user can also provide an input file and an output
   file.  The input file should be a wav file.  The output file will
   contain the convoluted input file.  If an input file is not provided,
   the program randomly generates data for the convolution.
*/
int main(int argc, char **argv) 
{
   int blocks = 0, threads = 0, badArgs = 0;
   char *inFile = NULL, *outFile = NULL;
   sndFileT snd;
   float *blurV = NULL; 

   //parse the command line arguments and make sure they
   //are good
   parseCmdLineArgs(argc, argv, &blocks, &threads, &inFile, &outFile);
   badArgs = checkArgs(blocks, threads, inFile, outFile);
   if (badArgs) verbose();

   snd.sndFile = (inFile != NULL) && (outFile != NULL);

   //generate the blur vector
   blurV = generateBlurVector(); 

   //if a user provided a sound file, read the data
   //and fill the snd sndFileT struct
   if (snd.sndFile) getSoundData(&snd, inFile, outFile);

   //perform the convolution on the CPU and the GPU
   gaussianTests(&snd, blurV, blocks, threads);

   //free the blurV and reset the GPU
   free(blurV);
   CHECK(cudaDeviceReset());
}


