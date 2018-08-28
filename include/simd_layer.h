#ifndef SIMD_LAYER_H
#define SIMD_LAYER_H

#include <memory>
#include "tool.h"
#include <assert.h>
#include <cstring>

// default 3x3
// /const int KERNEL_SIZE = 3;

namespace SIMD_KERNEL
{

	enum SIMD_ALG {
		SIMD_1_TILE_PER_ITER = 0,
		SIMD_2_TILE_PER_ITER,
	};

	template <typename Dtype>
	class SimdLayer {

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

		SIMD_ALG m_alg;

	private:

		Dtype* m_output;

	public:

		SimdLayer(SIMD_ALG alg, int batch_size, int iH, int iW, int iC, int kH, int kW, int sH, int sW, int oC, int pad, bool bias = true): m_alg(alg) {

			//assert(kH == kW, "kernel 3x3 is the best choice, some errors may occur for other kernels");
			assert(kH == kW);
			assert(pad == 0);
			assert(sW == 1);
			assert(sH == 1);
			assert(kH == 3);
			assert(kW == 3);

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

		void simd_1_tile_alg(Dtype* data, const Dtype* par, Dtype* col_buff) {

			const Dtype* kernel = par;

			for (int p=0; p<conv_out_channels_; p++)
			{
				Dtype* out = &m_output[p*m_oH*m_oW];

				for (int q=0; q<conv_in_channels_; q++)
				{
					Dtype* outptr = out;

					const Dtype* img0 = &data[q*m_iH*m_iW];

					const Dtype* kernel0 = kernel + p*conv_in_channels_*9  + q*9;

					const Dtype* r0 = img0;
					const Dtype* r1 = img0 + m_iW;
					const Dtype* r2 = img0 + m_iW*2;

					const Dtype* k0 = kernel0;
					const Dtype* k1 = kernel0 + 3;
					const Dtype* k2 = kernel0 + 6;

					int i = 0;

					for (; i < m_oH; i++)
					{
						int remain = m_oW;

						for (; remain>0; remain--)
						{
							Dtype sum = 0;

							sum += r0[0] * k0[0];
							sum += r0[1] * k0[1];
							sum += r0[2] * k0[2];
							sum += r1[0] * k1[0];
							sum += r1[1] * k1[1];
							sum += r1[2] * k1[2];
							sum += r2[0] * k2[0];
							sum += r2[1] * k2[1];
							sum += r2[2] * k2[2];

							*outptr += sum;

							r0++;
							r1++;
							r2++;
							outptr++;
						}

						r0 += 2;
						r1 += 2;
						r2 += 2;
					}

				}
			}
		}


		void simd_2_tile_alg(Dtype* data, const Dtype* par, Dtype* col_buff) {

			const Dtype* kernel = par;

			for (int p=0; p<conv_out_channels_; p++)
			{
				Dtype* out = &m_output[p*m_oH*m_oW];

				for (int q=0; q<conv_in_channels_; q++)
				{
					Dtype* outptr = out;
					Dtype* outptr2 = outptr + m_oW;

					const Dtype* img0 = &data[q*m_iH*m_iW];

					const Dtype* kernel0 = kernel + p*conv_in_channels_*9  + q*9;

					const Dtype* r0 = img0;
					const Dtype* r1 = img0 + m_iW;
					const Dtype* r2 = img0 + m_iW*2;
					const Dtype* r3 = img0 + m_iW*3;

					const Dtype* k0 = kernel0;
					const Dtype* k1 = kernel0 + 3;
					const Dtype* k2 = kernel0 + 6;

					int i = 0;

					for (; i+1 < m_oH; i+=2)
					{

						int remain = m_oW;

						for (; remain>0; remain--)
						{
							Dtype sum = 0;
							Dtype sum2 = 0;

							sum += r0[0] * k0[0];
							sum += r0[1] * k0[1];
							sum += r0[2] * k0[2];
							sum += r1[0] * k1[0];
							sum += r1[1] * k1[1];
							sum += r1[2] * k1[2];
							sum += r2[0] * k2[0];
							sum += r2[1] * k2[1];
							sum += r2[2] * k2[2];

							sum2 += r1[0] * k0[0];
							sum2 += r1[1] * k0[1];
							sum2 += r1[2] * k0[2];
							sum2 += r2[0] * k1[0];
							sum2 += r2[1] * k1[1];
							sum2 += r2[2] * k1[2];
							sum2 += r3[0] * k2[0];
							sum2 += r3[1] * k2[1];
							sum2 += r3[2] * k2[2];

							*outptr += sum;
							*outptr2 += sum2;

							r0++;
							r1++;
							r2++;
							r3++;
							outptr++;
							outptr2++;
						}

						r0 += 2 + m_iW;
						r1 += 2 + m_iW;
						r2 += 2 + m_iW;
						r3 += 2 + m_iW;

						outptr += m_oW;
						outptr2 += m_oW;
					}

					for (; i < m_oH; i++)
					{
						int remain = m_oW;

						for (; remain>0; remain--)
						{
							Dtype sum = 0;

							sum += r0[0] * k0[0];
							sum += r0[1] * k0[1];
							sum += r0[2] * k0[2];
							sum += r1[0] * k1[0];
							sum += r1[1] * k1[1];
							sum += r1[2] * k1[2];
							sum += r2[0] * k2[0];
							sum += r2[1] * k2[1];
							sum += r2[2] * k2[2];

							*outptr += sum;

							r0++;
							r1++;
							r2++;
							outptr++;
						}

						r0 += 2;
						r1 += 2;
						r2 += 2;
					}

				}
			}
		}

		const Dtype* get_inference_cpu(Dtype* data, const Dtype* par, Dtype* col_buff) {

			switch (m_alg) {
			case SIMD_1_TILE_PER_ITER:
				simd_1_tile_alg(data, par, col_buff); break;
			case SIMD_2_TILE_PER_ITER:
				simd_2_tile_alg(data, par, col_buff); break;
				break;
			}
			return  m_output;
		}

		void clear() {
			memset(m_output, 0, sizeof(Dtype)*m_oH*m_oW*conv_out_channels_);
		}


	public:
		~SimdLayer() {
		}


	private:

		

	};
}

#endif