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
    if (argc != 4) {
        fprintf(stderr, "Invalid Number of Arguments!\nUsage:\n");
        fprintf(stderr, "<Executable Name> <input image path> <threshold> <max_value> \n");
        return -1;
    }

    cv::Mat in_img, out_img, ocv_thresh, ocv_erode, bw_img;

    unsigned short in_width, in_height;

    /*  reading in the color image  */
    in_img = cv::imread(argv[1], 0);

    std::cout << argc << std::endl;

    if (in_img.data == NULL) {
        fprintf(stderr, "Cannot open image at %s\n", argv[1]);
        return 0;
    }

    in_width = in_img.cols;
    in_height = in_img.rows;

    out_img.create(in_img.rows, in_img.cols, in_img.depth());
    ocv_thresh.create(in_img.rows, in_img.cols, in_img.depth());
    ocv_erode.create(in_img.rows, in_img.cols, in_img.depth());
    bw_img.create(in_img.rows, in_img.cols, in_img.depth());

    cv::bitwise_not(in_img,bw_img);

    imwrite("bw_img.jpg", bw_img);

    unsigned char thresh = atoi(argv[2]);
    unsigned char maxval = atoi(argv[3]);
    fprintf(stdout, "Threshold value: %d Maximum value: %d\n", int(thresh), int(maxval));

    cv::Mat element = cv::getStructuringElement(KERNEL_SHAPE, cv::Size(FILTER_SIZE, FILTER_SIZE), cv::Point(-1, -1));

    std::vector<unsigned char> shape(FILTER_SIZE * FILTER_SIZE);

    for (int i = 0; i < (FILTER_SIZE * FILTER_SIZE); i++) {
        shape[i] = element.data[i];
    }

    size_t vec_in_size_bytes = FILTER_SIZE * FILTER_SIZE * sizeof(unsigned char);
    //size_t vec_in_size_bytes = FILTER_SIZE * FILTER_SIZE;

    cv::threshold(bw_img, ocv_thresh, thresh, maxval, THRESH_TYPE);
    imwrite("thresh_img.jpg", ocv_thresh);
    cv::erode(ocv_thresh, ocv_erode, element);
    imwrite("erode_img.jpg", ocv_erode);

    /////////////////////////////////////// CL ////////////////////////

    int height = bw_img.rows;
    int width = bw_img.cols;

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


    // Set the kernel arguments
    OCL_CHECK(err, err = kernel.setArg(0, imageToDevice));
    OCL_CHECK(err, err = kernel.setArg(1, buffer_inShape));
    OCL_CHECK(err, err = kernel.setArg(2, imageFromDevice));
    OCL_CHECK(err, err = kernel.setArg(3, height));
    OCL_CHECK(err, err = kernel.setArg(4, width));
    OCL_CHECK(err, err = kernel.setArg(5, thresh));
    OCL_CHECK(err, err = kernel.setArg(6, maxval));

    cl::Event event;
    OCL_CHECK(err, q.enqueueWriteBuffer(imageToDevice, CL_TRUE, 0, (height * width), bw_img.data));
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
    ocv_thresh.~Mat();
    ocv_erode.~Mat();

    return 0;
}
