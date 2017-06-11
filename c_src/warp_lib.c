#include <TH/TH.h>
#include <stdio.h>

typedef unsigned char byte;

int warpForward(THFloatTensor *inputImage, THFloatTensor *flow, THFloatTensor *recImage)
{
    // size
    const int BATCH = inputImage->size[0];
    const int HEIGHT = inputImage->size[2];
    const int WIDTH = inputImage->size[3];

    for (int b=0; b < BATCH; ++b) {
        for (int i=0; i < HEIGHT; ++i) {
            for (int j=0; j < WIDTH; ++j) {

                float i2 = i + THTensor_fastGet4d(flow, b, 0, i, j);
                float j2 = j + THTensor_fastGet4d(flow, b, 1, i, j);

                int floor_i2 = (int)floor(i2);
                int floor_j2 = (int)floor(j2);
                int ceil_i2 = (int)ceil(i2);
                int ceil_j2 = (int)ceil(j2);

                // if not in rec image
                if (!(floor_i2 >= 0 &&
                      floor_j2 >= 0 &&
                      ceil_i2 < HEIGHT &&
                      ceil_j2 < WIDTH)) {
                      for (int c=0; c<3; c++) {
                           THTensor_fastSet4d(recImage, b, c, i, j, 0);
                      }
                      continue;
                }

                float theta_x = i2 - floor_i2;
                float theta_y = j2 - floor_j2;

                for (int c=0; c<3; ++c) {
                    float value = (1 - theta_x) * (1 - theta_y) * THTensor_fastGet4d(inputImage, b, c, floor_i2, floor_j2) +
                                  theta_x * (1 - theta_y) * THTensor_fastGet4d(inputImage, b, c, ceil_i2, floor_j2) +
                                  (1 - theta_x) * theta_y * THTensor_fastGet4d(inputImage, b, c, floor_i2, ceil_j2) +
                                  theta_x * theta_y * THTensor_fastGet4d(inputImage, b, c, ceil_i2, ceil_j2);
                    byte final_value = (byte)floor(value);
                    THTensor_fastSet4d(recImage, b, c, i, j, final_value);
                }
            }
        }
    }
    return 1;
}

int warpBackward(THFloatTensor *inputImage, THFloatTensor *flow, THFloatTensor *gradFlow, THFloatTensor *gradOutput)
{
    // size
    const int BATCH = inputImage->size[0];
    const int HEIGHT = inputImage->size[2];
    const int WIDTH = inputImage->size[3];

    for (int b=0; b < BATCH; ++b) {
        for (int i=0; i < HEIGHT; ++i) {
            for (int j=0; j < WIDTH; ++j) {

                float i2 = i + THTensor_fastGet4d(flow, b, 0, i, j);
                float j2 = j + THTensor_fastGet4d(flow, b, 1, i, j);

                int floor_i2 = (int)floor(i2);
                int floor_j2 = (int)floor(j2);
                int ceil_i2 = (int)ceil(i2);
                int ceil_j2 = (int)ceil(j2);

                // if not in grad feature map
                if (!(floor_i2 >= 0 &&
                      floor_j2 >= 0 &&
                      ceil_i2 < HEIGHT &&
                      ceil_j2 < WIDTH)) {
                      THTensor_fastSet4d(gradFlow, b, 0, i, j, 0);
                      THTensor_fastSet4d(gradFlow, b, 1, i, j, 0);
                      continue;
                }

                float theta_x = i2 - floor_i2;
                float theta_y = j2 - floor_j2;

                float x_grad = 0.0;
                for (int c=0; c < 3; ++c) {
                     float value = (theta_y - 1) * THTensor_fastGet4d(inputImage, b, c, floor_i2, floor_j2) +
                                  (1 - theta_y) * THTensor_fastGet4d(inputImage, b, c, ceil_i2, floor_j2) +
                                  (-theta_y) * THTensor_fastGet4d(inputImage, b, c, floor_i2, ceil_j2) +
                                  theta_y * THTensor_fastGet4d(inputImage, b, c, ceil_i2, ceil_j2);
                     x_grad += value * THTensor_fastGet4d(gradOutput, b, c, i, j);
                }
                THTensor_fastSet4d(gradFlow, b, 0, i, j, x_grad / 3.0);

                float y_grad = 0.0;
                for (int c=0; c < 3; ++c) {
                   float value = (theta_x - 1) * THTensor_fastGet4d(inputImage, b, 1, floor_i2, floor_j2) +
                                  (1 - theta_x) * THTensor_fastGet4d(inputImage, b, 1, ceil_i2, floor_j2) +
                                  (-theta_x) * THTensor_fastGet4d(inputImage, b, 1, floor_i2, ceil_j2) +
                                  theta_x * THTensor_fastGet4d(inputImage, b, 1, ceil_i2, ceil_j2);
                    y_grad += value * THTensor_fastGet4d(gradOutput, b, c, i, j);
                }
                THTensor_fastSet4d(gradFlow, b, 1, i, j, y_grad / 3.0);
            }
        }
    }

    return 1;
}