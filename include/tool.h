#ifndef PUBLIC_TOOL_H
#define PUBLIC_TOOL_H

#include <fstream>
#include <iostream>
#include "mathlib.h"
#include <assert.h>
#include <sys/time.h>

namespace PUBLIC_TOOL{

	template<typename Dtype>
	Dtype max(Dtype a, Dtype b) {
		if (a > b) return a;
		else return b;
	}

	template<typename Dtype>
	Dtype min(Dtype a, Dtype b) {
		if (a < b) return a;
		else return b;
	}

	 void dlm_cpu_gemm(const CBLAS_TRANSPOSE TransA,
		const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		const float alpha, const float* A, const float* B, const float beta,
		float* C) {
		int lda = (TransA == CblasNoTrans) ? K : M;
		int ldb = (TransB == CblasNoTrans) ? N : K;
		cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
			ldb, beta, C, N);
	}
	
	 void dlm_cpu_gemm(const CBLAS_TRANSPOSE TransA,
		 const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
		 const double alpha, const double* A, const double* B, const double beta,
		 double* C) {
		 int lda = (TransA == CblasNoTrans) ? K : M;
		 int ldb = (TransB == CblasNoTrans) ? N : K;
		 cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
			 ldb, beta, C, N);
	 }

	 void print_tensor(const float* ptr, const int N, const int C, const int H, const int W, const char* strings) {
		 assert(N == 1);

		 for(int c_idx = 0; c_idx < C; c_idx++) {
		 	printf("%s channel %d\n", strings, c_idx);
			for(int h_idx = 0; h_idx < H; h_idx++) {
				for(int w_idx = 0; w_idx < W; w_idx++) {
					printf("%f ", ptr[(c_idx*H + h_idx)*W + w_idx]);
				}
				printf("\n");
			}
		 }
	 }

	double get_current_time()
	{
		struct timeval tv;
		gettimeofday(&tv, NULL);
		return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
	}

	void benchmark(const char* strings, double start, double end, int loop_num)
  	{
      printf("\n\n\n%-20s %8.2lfms", strings,  (end - start)/loop_num);
      printf("\n");
  	}


};

#endif			