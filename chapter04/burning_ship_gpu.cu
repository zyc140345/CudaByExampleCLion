//
// Created by 张易诚 on 24-9-15.
//

#include "../common/book.h"
#include "../common/cpu_bitmap.h"

#define DIM 1000
#define MAX_ITER 100

__device__ int burningShip(int x, int y) {
    const float scale = 1.7;
    float scaled_x = scale * ((float) (x - DIM / 2) / (DIM / 2) - 0.25);
    float scaled_y = scale * ((float) (DIM / 2 - y) / (DIM / 2) - 0.15);

    float zx = scaled_x;
    float zy = scaled_y;

    int i;
    for (i = 0; i < MAX_ITER; i++) {
        float xtemp = zx * zx - zy * zy + scaled_x;
        zy = std::abs(2 * zx * zy) + scaled_y;
        zx = xtemp;
        if (zx * zx + zy * zy >= 4)
            break;
    }

    return i;
}

__device__ float iter_to_magnitude(int iter) {
    return sqrtf(1 - pow(iter / (float) MAX_ITER - 1, 2));
}

__global__ void kernel(unsigned char *ptr) {
    // map from blockIdx to pixel position
    int x = blockIdx.x;
    int y = blockIdx.y;
    int offset = x + y * gridDim.x;

    // now calculate the value at that position
    int iter = burningShip(x, y);
    if (iter == MAX_ITER) {
        ptr[offset * 4 + 0] = 0;
        ptr[offset * 4 + 1] = 0;
        ptr[offset * 4 + 2] = 0;
        ptr[offset * 4 + 3] = 255;
    } else {
        auto redValue = (unsigned char) (iter_to_magnitude(iter) * (255 - 82) * 82 / 115);
        auto greenValue = (unsigned char) (iter_to_magnitude(iter) * (255 - 115));
        auto blueValue = (unsigned char) (iter_to_magnitude(iter) * (255 - 37) * 37 / 115);
        ptr[offset * 4 + 0] = 82 + redValue;
        ptr[offset * 4 + 1] = 115 + greenValue;
        ptr[offset * 4 + 2] = 37 + blueValue;
        ptr[offset * 4 + 3] = 255;
    }
}

// globals needed by the update routine
struct DataBlock {
    unsigned char *dev_bitmap;
};

int main(void) {
    DataBlock data;
    CPUBitmap bitmap(DIM, DIM, &data);
    unsigned char *dev_bitmap;

    HANDLE_ERROR(cudaMalloc((void **) &dev_bitmap, bitmap.image_size()));
    data.dev_bitmap = dev_bitmap;

    dim3 grid(DIM, DIM);
    kernel<<<grid, 1>>>(dev_bitmap);

    HANDLE_ERROR(cudaMemcpy(bitmap.get_ptr(), dev_bitmap,
                            bitmap.image_size(),
                            cudaMemcpyDeviceToHost));

    HANDLE_ERROR(cudaFree(dev_bitmap));

    bitmap.display_and_exit();
}
