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

#ifndef _XF_THRESHOLD_CONFIG_H_
#define _XF_THRESHOLD_CONFIG_H_

#include "hls_stream.h"
#include "ap_int.h"

#include "common/xf_common.hpp"
#include "common/xf_utility.hpp"

#include "imgproc/xf_threshold.hpp"
#include "imgproc/xf_erosion.hpp"
#include "imgproc/xf_dilation.hpp"
#include "xf_config_params.h"

typedef ap_uint<8> ap_uint8_t;
typedef ap_uint<64> ap_uint64_t;

/*  set the height and weight  */
#define HEIGHT 2160
#define WIDTH 3840

#if RO
#define NPIX XF_NPPC8
#endif
#if NO
#define NPIX XF_NPPC1
#endif

// Resolve optimization type:
#if NO
#define NPC1 XF_NPPC1
#if GRAY
#define PTR_WIDTH 8
#else
#define PTR_WIDTH 32
#endif
#endif

#if RO
#define NPC1 XF_NPPC8
#if GRAY
#define PTR_WIDTH 64
#else
#define PTR_WIDTH 256
#endif
#endif
// Set pixel depth:
#if GRAY
#define TYPE XF_8UC1
#else
#define TYPE XF_8UC3
#endif
/* config width and height */
#define WIDTH 3840
#define HEIGHT 2160

/*void medimg_accel(xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX>& _src,
                     xf::cv::Mat<XF_8UC1, HEIGHT, WIDTH, NPIX>& _dst,
                     unsigned char thresh,
                     unsigned char maxval);*/

#endif // end of _XF_THRESHOLD_CONFIG_H_
