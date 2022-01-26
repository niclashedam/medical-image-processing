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

#include "common/xf_headers.hpp"
#include "xf_canny_config.h"

#include "xcl2.hpp"
#include <time.h>

typedef unsigned char NMSTYPE;

int main(int argc, char** argv) {
    //# Images
    cv::Mat in_img;
    cv::Mat img_gray;

    if (argc != 2) {
        fprintf(stderr, "Usage: <executable> <input image>\n");
        return -1;
    }

    in_img = cv::imread(argv[1], 1); // reading in the color image
    if (!in_img.data) {
        fprintf(stderr, "Failed to load the image ... %s\n ", argv[1]);
        return -1;
    }

    extractChannel(in_img, img_gray, 1); // Extract gray scale images

    imwrite("gray.png", img_gray); // Save HLS result

    in_img.~Mat();
    img_gray.~Mat();

    return 0;
}
