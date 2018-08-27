#ifndef PUBLIC_TOOL_H
#define PUBLIC_TOOL_H

#include <fstream>
#include <iostream>
#include "mathlib.h"
#include <assert.h>

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

	 void printTensor(const float* ptr, const int N, const int C, const int H, const int W, const char* strings) {
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


};

#endif			