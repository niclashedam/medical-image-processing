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
#include "xcl2.hpp"
#include <time.h>
#include "medimg_config.h"
#include <iostream>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> \n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_ref;

    unsigned short in_width, in_height;

    /*  reading in the color image  */
    in_img = cv::imread(argv[1], 0);

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    in_width = in_img.cols;
    in_height = in_img.rows;

    out_img.create(in_img.rows, in_img.cols, in_img.depth());

    imwrite("bw_img.jpg", in_img);


    unsigned char thresh = 100;
    unsigned char maxval = 50;

    cv::Mat element = cv::getStructuringElement(KERNEL_SHAPE, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));

    std::vector<unsigned char> shape(FILTER_SIZE * FILTER_SIZE);

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        shape[i] = element.data[i];
    }

    size_t vec_in_size_bytes = FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char);
    //size_t vec_in_size_bytes = FILTER_SIZE * FILTER_SIZE;

    /////////////////////////////////////// CL ////////////////////////

    int height = in_img.rows;
    int width = in_img.cols;

    cl_int err;
    std::cout << "INFO: Running OpenCL section." << std::endl;

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_medimg");
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);

    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

    // Create a kernel:
    OCL_CHECK(err, cl::Kernel kernel(program, "medimg_accel", &err));

    std::vector<cl::Memory> inBufVec, outBufVec;
    OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width), NULL, &err));
    OCL_CHECK(err, cl::Buffer buffer_inShape(context, CL_MEM_READ_ONLY, vec_in_size_bytes, NULL, &err));
    OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, (height * width), NULL, &err));
    //OCL_CHECK(err, cl::Buffer buffer_inShape(context, CL_MEM_READ_ONLY, vec_in_size_bytes, NULL, &err));

    // Set the kernel arguments
    OCL_CHECK(err, err = kernel.setArg(0, imageToDevice));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inShape));
    OCL_CHECK(err, err = kernel.setArg(2, imageFromDevice));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));
    OCL_CHECK(err, err = kernel.setArg(5, thresh));
    OCL_CHECK(err, err = kernel.setArg(6, maxval));

    cl::Event event;
    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height * width), in_img.data));
    //OCL_CHECK(err, q.enqueueWriteBuffer(buffer_inShape, CL_TRUE, 0, vec_in_size_bytes, shape.data()));
    OCL_CHECK(err, q.enqueueWriteBuffer(buffer_inShape, CL_TRUE, 0, vec_in_size_bytes, shape.data(), nullptr, &event));

    // Profiling Objects
    cl_ulong start = 0;
    cl_ulong end = 0;
    double diff_prof = 0.0f;
    cl::Event event_sp;

    // Launch the kernel
    OCL_CHECK(err, err = q.enqueueTask(kernel, NULL, &event_sp));
    clWaitForEvents(1, (const cl_event*)&event_sp);

    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
    event_sp.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
    diff_prof = end - start;
    std::cout << (diff_prof / 1000000) << "ms" << std::endl;

    // Copying Device result data to Host memory
    q.enqueueReadBuffer(imageFromDevice, CL_TRUE, 0, (height * width), out_img.data);

    q.finish();

    /////////////////////////////////////// end of CL ////////////////////////

    // Write output image
    imwrite("hls_out.jpg", out_img);

    in_img.~Mat();
    out_img.~Mat();

    return 0;
}

/*typedef unsigned char NMSTYPE;

int main(int argc, char** argv) {
    //# Images
    cv::Mat in_img, out_img;
    cv::Mat img_gray;

    int height, width;
    int low_threshold, high_threshold;
    height = img_gray.rows;
    width = img_gray.cols;

    //low_threshold = 30;
    //high_threshold = 64;
    unsigned char maxval = 50;
    unsigned char thresh = 100;

    std::cout << argv[1] << " " << argv[0] << std::endl;

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
    cl_int err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    //cl::Context context(device);
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
    // Create command queue
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

    // Create command queue
    //cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

    // Load kernel unto device
    std::string device_name = device.getInfo<CL_DEVICE_NAME>();
    std::string binaryFile = xcl::find_binary_file(device_name, "krnl_medimg");
    //std::string binaryFile = argv[0];
    cl::Program::Binaries bins = xcl::import_binary_file(binaryFile);
    devices.resize(1);


    std::cout << "kernel init" << std::endl;
    // Initialize local program
    //cl::Program program(context, devices, bins);
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));
    //cl::Kernel krnl(program, "medimg_accel");
    OCL_CHECK(err, cl::Kernel krnl(program, "medimg_accel", &err));

    std::cout << "define memory buffers" << std::endl;
    // Define memory buffers for input and ouput image
    std::vector<cl::Memory> inBufVec, outBufVec;
    cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width));
    cl::Buffer imageFromDevice(context, CL_MEM_READ_WRITE, (height * width));
    //OCL_CHECK(err, cl::Buffer imageToDevice(context, CL_MEM_READ_ONLY, (height * width), NULL, &err));
    //OCL_CHECK(err, cl::Buffer imageFromDevice(context, CL_MEM_WRITE_ONLY, (height * width), NULL, &err));

    // Set the kernel arguments
    krnl.setArg(0, imageToDevice);
    krnl.setArg(1, imageFromDevice);
    krnl.setArg(2, thresh);
    krnl.setArg(3, maxval);
    krnl.setArg(4, height);
    krnl.setArg(5, width);

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
}*/
