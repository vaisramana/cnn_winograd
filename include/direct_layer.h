#ifndef DIRECT_LAYER_H
#define DIRECT_LAYER_H

#include <memory>
#include "tool.h"
#include <assert.h>

// default 3x3
// /const int KERNEL_SIZE = 3;

namespace DIRECT_KERNEL
{

	template <typename Dtype>
	class DirectLayer {

	private:

		int m_group_;
		int m_batchSize;

		int m_bottom_dim_;// par size
		int m_top_dim_;

		// The following variables are initialized in WeightAlign
		int tile_h_in_, tile_w_in_; /* input tile size */
		int tile_h_out_, tile_w_out_; /* output tile size */
		int ntiles_h_, ntiles_w_; /* number of tiles */

		int conv_in_channels_; //ic
		int conv_out_channels_;//oc

		int m_iH;
		int m_iW;

		int m_oH;
		int m_oW;

		int m_kH;
		int m_kW;
		int m_sH;
		int m_sW;

		int m_pad;
		bool m_bias;

	private:

		Dtype* m_output;

	public:

		DirectLayer(int batch_size, int iH, int iW, int iC, int kH, int kW, int sH, int sW, int oC, int pad, bool bias = true) {

			//assert(kH == kW, "kernel 3x3 is the best choice, some errors may occur for other kernels");
			assert(kH == kW);
			assert(pad == 0);
			assert(sW == 1);
			assert(sH == 1);

			m_iH = iH;
			m_iW = iW;
			conv_in_channels_ = iC;
			m_kH = kH;
			m_kW = kW;
			m_sH = sH;
			m_sW = sW;
			conv_out_channels_ = oC;
			m_pad = pad; // pad_h = pad_w
			m_bias = bias;

			m_batchSize = batch_size;
			m_group_ = 1;

			m_bottom_dim_ = 0;// default batch =1
			m_top_dim_ = 0;

			// Output width.
			m_oW = (m_iW + m_pad * 2 - m_kW) / m_sW + 1;
			m_oH = (m_iH + m_pad * 2 - m_kH) / m_sH + 1;

			m_output = new Dtype [m_oH*m_oW*conv_out_channels_];

		}

		//template <typename Dtype>
		const Dtype* get_inference_cpu(Dtype* data, const Dtype* par, Dtype* col_buff) {

			/*
			printf("input\n");
			for(int inW = 0; inW < m_iW; inW++) {
				for(int inH = 0; inH < m_iH; inH++) {
					printf("%f ", data[inH*m_iW + inW]);
				}
				printf("\n");
			}

			printf("kernel\n");
			for(int inW = 0; inW < m_kW; inW++) {
				for(int inH = 0; inH < m_kH; inH++) {
					printf("%f ", par[inH*m_kW + inW]);
				}
				printf("\n");
			}
			*/
			//printf("m_oH %d m_oW %d conv_out_channels_ %d\n", m_oH, m_oW, conv_out_channels_);
			
			for (int batch = 0; batch < m_batchSize; batch++) {
				for (int out_y = 0; out_y < m_oH; out_y++) {
					for (int out_x = 0; out_x < m_oW; out_x++) {
						for (int out_channel = 0; out_channel < conv_out_channels_; out_channel++) {
							//const int in_x_origin = (out_x * m_sW) - m_pad;
          					//const int in_y_origin = (out_y * m_sH) - m_pad;
							float total = 0.f;
							for (int filter_y = 0; filter_y < m_kH; filter_y++) {
								for (int filter_x = 0; filter_x < m_kW; filter_x++) {
									for (int in_channel = 0; in_channel < conv_in_channels_; ++in_channel) {
										//const int in_x = in_x_origin + dilation_width_factor * filter_x;
										//const int in_y = in_y_origin + dilation_height_factor * filter_y;
                						const int in_x = out_x + filter_x;
										const int in_y = out_y + filter_y;
										if ((in_x >= 0) && (in_x < m_iW) && (in_y >= 0) && (in_y < m_iH)) {
											//NCHW
											//printf("fetch data[%d] filter[%d]\n", 
											//		(in_channel*m_iH + in_y)*m_iW + in_x,
											//		((out_channel*conv_in_channels_ + in_channel) *m_kW + filter_y) *m_kW + filter_x);
											float input_value  = data[(in_channel*m_iH + in_y)*m_iW + in_x];
											float filter_value = par[((out_channel*conv_in_channels_ + in_channel) *m_kW + filter_y) *m_kW + filter_x];
											total += (input_value * filter_value);
										}
									}
								}
							}
							//printf("output[%d]=%f \n", (out_channel*m_oH + out_y)*m_oW + out_x, total);
							m_output[(out_channel*m_oH + out_y)*m_oW + out_x] = total;
						}
					}
				}
			}
			return  m_output;
		}


	public:
		~DirectLayer() {
		}


	private:

		

	};
}

#endif