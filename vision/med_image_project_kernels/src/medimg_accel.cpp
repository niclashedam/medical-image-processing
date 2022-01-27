/*
 * Copyright 2019 Xilinx, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "medimg_config.h"

extern "C" {
void medimg_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
		unsigned char* process_shape,
		ap_uint<OUTPUT_PTR_WIDTH>* img_out,
		int rows,
		int cols,
		unsigned char thresh,
		unsigned char maxval) {
    #pragma HLS INTERFACE m_axi     port=img_inp  offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi     port=process_shape offset=slave  bundle=gmem1
    #pragma HLS INTERFACE m_axi     port=img_out  offset=slave bundle=gmem2

    #pragma HLS INTERFACE s_axilite port=rows
    #pragma HLS INTERFACE s_axilite port=cols
	#pragma HLS INTERFACE s_axilite port=thresh
    #pragma HLS INTERFACE s_axilite port=maxval
    #pragma HLS INTERFACE s_axilite port=return

    const int pROWS = HEIGHT;
    const int pCOLS = WIDTH;
    const int pNPC1 = NPIX;

    // Copy the shape data:
    unsigned char _kernel[FILTER_SIZE * FILTER_SIZE];
    for (unsigned int i = 0; i < FILTER_SIZE * FILTER_SIZE; ++i) {
        #pragma HLS PIPELINE
        _kernel[i] = process_shape[i];
    }

    //xf::cv::Mat<TYPE, HEIGHT, WIDTH, NPC1> thresholdOut(height, width);

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX> in_mat(rows, cols);
    #pragma HLS stream variable=in_mat.data depth=2

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX> out_mat(rows, cols);
    #pragma HLS stream variable=out_mat.data depth=2

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX> threshold_out(rows, cols);
	#pragma HLS stream variable=threshold_out.data depth=2

    xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX> erode_out(rows, cols);
	#pragma HLS stream variable=erode_out.data depth=2

    #pragma HLS DATAFLOW

    xf::cv::Array2xfMat<INPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPIX>(img_inp, in_mat);

    xf::cv::Threshold<THRESH_TYPE, XF_8UC1, HEIGHT, WIDTH, NPIX>(in_mat, threshold_out, thresh, maxval);

    xf::cv::erode<XF_BORDER_CONSTANT, TYPE, HEIGHT, WIDTH, KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS, NPC1>(threshold_out, erode_out, _kernel);

    xf::cv::dilate<XF_BORDER_CONSTANT, TYPE, HEIGHT, WIDTH, KERNEL_SHAPE, FILTER_SIZE, FILTER_SIZE, ITERATIONS, NPC1>(erode_out, out_mat, _kernel);

    xf::cv::xfMat2Array<OUTPUT_PTR_WIDTH, XF_8UC1, HEIGHT, WIDTH, NPIX>(out_mat, img_out);
}
}
