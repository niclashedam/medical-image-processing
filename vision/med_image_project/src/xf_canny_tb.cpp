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
    cv::Mat in_img, out_img;
    cv::Mat img_gray;

    int height, width;
    int low_threshold, high_threshold;
    height = img_gray.rows;
    width = img_gray.cols;

    low_threshold = 30;
    high_threshold = 64;

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

    // Get a list of all devices and pick the first Xilinx device
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);

    // Create command queue
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load kernel unto device
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_canny");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);

    // Initialize local program
    cl::Program program(context, devices, bins);
    cl::Kernel krnl(program, "canny_accel");

    // Define memory buffers for input and ouput image
    std::vector<cl::Memory> inBufVec, outBufVec;
    cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width));
    cl::Buffer imageFromDevice(context, CL_MEM_READ_WRITE, (height * width));

    // Set the kernel arguments
    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
    krnl.setArg(2, height);
    krnl.setArg(3, width);
    krnl.setArg(4, low_threshold);
    krnl.setArg(5, high_threshold);

    // Enqueue command
    q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height * (width)), img_gray.data);

    // Launch the kernel and wait for completion
    cl::Event event_sp;
    q.enqueueTask(krnl, NULL, &event_sp);
    clWaitForEvents(1, (const cl_event*)&event_sp);

    // Copying Device result data to Host memory
    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, (height * width), out_img.data);

    q.finish();

    in_img.~Mat();
    img_gray.~Mat();
    out_img.~Mat();

    return 0;
}
