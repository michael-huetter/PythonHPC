import numpy as np
import matplotlib.pyplot as plt
import time

from numba import jit, cuda

# Mandelbrot parameters
width = 2000
height = 2000
max_iter = 200
x_min, x_max = -2.0, 1.0
y_min, y_max = -1.5, 1.5

#---------------------------------------
# CPU VERSION (Numba JIT without parallelization)
#---------------------------------------
@jit(nopython=True)
def mandelbrot_cpu(max_iter, x_min, x_max, y_min, y_max, width, height):
    image = np.zeros((height, width), dtype=np.uint16)
    for i in range(height):
        y0 = y_min + (y_max - y_min) * i / height
        for j in range(width):
            x0 = x_min + (x_max - x_min) * j / width
            x, y = 0.0, 0.0
            iteration = 0
            while x*x + y*y <= 4.0 and iteration < max_iter:
                x_new = x*x - y*y + x0
                y = 2.0*x*y + y0
                x = x_new
                iteration += 1
            image[i, j] = iteration
    return image

#---------------------------------------
# GPU VERSION (Numba CUDA)
#---------------------------------------
@cuda.jit
def mandelbrot_cuda_kernel(max_iter, x_min, x_max, y_min, y_max, width, height, output):
    i, j = cuda.grid(2)
    if i < height and j < width:
        x0 = x_min + (x_max - x_min) * j / width
        y0 = y_min + (y_max - y_min) * i / height

        x, y = 0.0, 0.0
        iteration = 0
        while x*x + y*y <= 4.0 and iteration < max_iter:
            x_new = x*x - y*y + x0
            y = 2.0*x*y + y0
            x = x_new
            iteration += 1

        output[i, j] = iteration

def mandelbrot_gpu(max_iter, x_min, x_max, y_min, y_max, width, height):
    threadsperblock = (16, 16)
    blockspergrid_x = (width + threadsperblock[0] - 1) // threadsperblock[0]
    blockspergrid_y = (height + threadsperblock[1] - 1) // threadsperblock[1]
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    output = np.zeros((height, width), dtype=np.uint16)
    d_output = cuda.to_device(output)

    mandelbrot_cuda_kernel[blockspergrid, threadsperblock](max_iter, x_min, x_max, y_min, y_max, width, height, d_output)
    d_output.copy_to_host(output)
    return output

#---------------------------------------
# Run and time CPU version
#---------------------------------------
start_cpu = time.time()
image_cpu = mandelbrot_cpu(max_iter, x_min, x_max, y_min, y_max, width, height)
end_cpu = time.time()
cpu_time = end_cpu - start_cpu

print(f"CPU (Numba JIT) execution time: {cpu_time:.4f} seconds")

#---------------------------------------
# Run and time GPU version
#---------------------------------------
start_gpu = time.time()
image_gpu = mandelbrot_gpu(max_iter, x_min, x_max, y_min, y_max, width, height)
end_gpu = time.time()
gpu_time = end_gpu - start_gpu

print(f"GPU (Numba CUDA) execution time: {gpu_time:.4f} seconds")

#---------------------------------------
# Plot and save to PDF
#---------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(12,6))

ax1 = axs[0]
ax1.imshow(image_cpu, cmap="magma", extent=[x_min, x_max, y_min, y_max])
ax1.set_title("Mandelbrot (CPU)")

ax2 = axs[1]
ax2.imshow(image_gpu, cmap="magma", extent=[x_min, x_max, y_min, y_max])
ax2.set_title("Mandelbrot (GPU)")

plt.tight_layout()
plt.savefig("mandelbrot.pdf", dpi=300)
plt.show()
