#pragma once

#include "../include/winograd_kernel.h"
#include "../include/winograd_layer.h"
#include "../include/direct_layer.h"
#include "../include/simd_layer.h"
#include "../include/tool.h"
#include <iostream>
#include <iomanip>

using namespace WINOGRAD_KERNEL;
using namespace std;

const int CIN = 1;
const int COUT = 32;

const int IH = 128;
const int IW = 128;

const int PRECISE = 8;

#define INPUT_INTEGER 1
#define KERNEL_INTEGER 1
#define DATA_PRINT 0

void testWinograd(int inH, int inW, int inC, int outC, int loop_num);

int main(int argc, const char * argv[])  {


	WINOGRAD_KERNEL::winograd2D_initialize();

	if(argc == 6) {
		printf("profiling H %d W %d inC %d outC %d loop %d\n", 
				atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
		testWinograd(atoi(argv[1]), atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
	}
	else {
		testWinograd(IH, IW, CIN, COUT, 1);
	}
	

	return 0;
}

void testWinograd(int inH, int inW, int inC, int outC, int loop_num) {

	//int batch_size = 1;

	int tiH = inH;
	int tiW = inW;

	int tkW = 3;
	int tkH = 3;

	int tsW = 1;
	int tsH = 1;

	int tiC = inC;
	const int toC = outC;

	bool tbias = true;

	int tpad = 0;

	const auto toH = (tiH + tpad * 2 - tkH) / tsH + 1;

	// Output width.
	const auto toW = (tiW + tpad * 2 - tkW) / tsW + 1;

	cout << setprecision(PRECISE);

	//NCHW
	float* input = new float[tiC*tiH*tiW];
	float* kernel = new float[tiC*tkH*tkW*toC];

	//initInput
	for (int c = 0; c<tiC*tiH*tiW; ) {

#if INPUT_INTEGER
		input[c++] = rand() % 10;
#else
		input[c++] = rand()  * 0.1234f / RAND_MAX;//rand() % 10;//
#endif

	}

	//initKernel
	for(int c=0;c< tiC*tkH*tkW*toC;)
	{

#if KERNEL_INTEGER
			kernel[c++] = rand() % 10;//
#else 
			kernel[c++] = rand()*0.1234 / RAND_MAX; //
#endif		
		
	}


	WINOGRAD_KERNEL::WinogradLayer<float> wt8X8(
		WINOGRAD_KERNEL::WT_8X8_F_6X6_3X3, //WT_6X6_F_4X4_3X3
		1,
		tiH,
		tiW,
		tiC,
		tkH,
		tkW,
		tsH,
		tsW,
		toC,
		tpad,
		tbias
	);

	WINOGRAD_KERNEL::WinogradLayer<float> wt6x6(
		WINOGRAD_KERNEL::WT_6X6_F_4X4_3X3, //WT_6X6_F_4X4_3X3
		1,
		tiH,
		tiW,
		tiC,
		tkH,
		tkW,
		tsH,
		tsW,
		toC,
		tpad,
		tbias
	);

	DIRECT_KERNEL::DirectLayer<float> direct(
		1,
		tiH,
		tiW,
		tiC,
		tkH,
		tkW,
		tsH,
		tsW,
		toC,
		tpad,
		tbias
	);

	SIMD_KERNEL::SimdLayer<float> simd_2_tile(
		SIMD_KERNEL::SIMD_2_TILE_PER_ITER,
		1,
		tiH,
		tiW,
		tiC,
		tkH,
		tkW,
		tsH,
		tsW,
		toC,
		tpad,
		tbias
	);

	SIMD_KERNEL::SimdLayer<float> simd_1_tile(
		SIMD_KERNEL::SIMD_1_TILE_PER_ITER,
		1,
		tiH,
		tiW,
		tiC,
		tkH,
		tkW,
		tsH,
		tsW,
		toC,
		tpad,
		tbias
	);

#if DATA_PRINT
	PUBLIC_TOOL::print_tensor(input, 1, tiC, tiW, tiW, "input");
	PUBLIC_TOOL::print_tensor(kernel, 1, tiC, tkH, tkW, "kernel");
#endif

	float* buffer=new float [toH*toW*tiC*100];// enough buffer, used as medium buffer flowing through each layer

	double start, end;
	const float* output;

#ifndef __aarch64__
	//wt8x8
	start = PUBLIC_TOOL::get_current_time();
	for(int i=0; i<loop_num; i++) {
		output = wt8X8.get_inference_cpu(input, kernel, (float*)buffer);
	}
	end = PUBLIC_TOOL::get_current_time();
    PUBLIC_TOOL::benchmark("wt8x8", start, end, loop_num);
#if DATA_PRINT	
	PUBLIC_TOOL::print_tensor(output, 1, toC, toH, toW, "wt8x8");
#endif

	//wt6x6
	start = PUBLIC_TOOL::get_current_time();
	for(int i=0; i<loop_num; i++) {
		output = wt6x6.get_inference_cpu(input, kernel, (float*)buffer);
	}
	end = PUBLIC_TOOL::get_current_time();
    PUBLIC_TOOL::benchmark("wt6x6", start, end, loop_num);
#if DATA_PRINT	
	PUBLIC_TOOL::print_tensor(output, 1, toC, toH, toW, "wt6x6");
#endif
#endif

	//direct
	start = PUBLIC_TOOL::get_current_time();
	for(int i=0; i<loop_num; i++) {
		output = direct.get_inference_cpu(input, kernel, (float*)buffer);
	}
	end = PUBLIC_TOOL::get_current_time();
    PUBLIC_TOOL::benchmark("direct", start, end, loop_num);
#if DATA_PRINT	
	PUBLIC_TOOL::print_tensor(output, 1, toC, toH, toW, "direct");
#endif


	//simd_2_tile
	start = PUBLIC_TOOL::get_current_time();
	for(int i=0; i<loop_num; i++) {
		output = simd_2_tile.get_inference_cpu(input, kernel, (float*)buffer);
	}
	end = PUBLIC_TOOL::get_current_time();
    PUBLIC_TOOL::benchmark("simd_2_tile", start, end, loop_num);
	simd_2_tile.clear();
	output = simd_2_tile.get_inference_cpu(input, kernel, (float*)buffer);
#if DATA_PRINT	
	PUBLIC_TOOL::print_tensor(output, 1, toC, toH, toW, "simd_2_tile");
#endif



	//simd_1_tile
	start = PUBLIC_TOOL::get_current_time();
	for(int i=0; i<loop_num; i++) {
		output = simd_1_tile.get_inference_cpu(input, kernel, (float*)buffer);
	}
	end = PUBLIC_TOOL::get_current_time();
    PUBLIC_TOOL::benchmark("simd_1_tile", start, end, loop_num);
	simd_1_tile.clear();
	output = simd_1_tile.get_inference_cpu(input, kernel, (float*)buffer);
#if DATA_PRINT	
	PUBLIC_TOOL::print_tensor(output, 1, toC, toH, toW, "simd_1_tile");
#endif

	delete[] buffer; 
}
