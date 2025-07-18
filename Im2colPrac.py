import numpy as np
x = np.array(
    [
        [1,2,3,1,5],
        [3,-1,-2,3,1],
        [-4,1,3,1,2],
        [2,4,5,1,5]]
)


input_data = x # 데이터 하나 선택 (--> 2차원 배열임임)
stride = 1
padding = 0
filter_h = 3
filter_w = 3




output_h = np.int32((input_data.shape[0] + 2*padding - filter_h) / stride) + 1
output_w = np.int32((input_data.shape[1] + 2*padding - filter_w) / stride) + 1

im2col = np.zeros((output_h*output_w, filter_h*filter_w))
for i in range(output_h) :
    for j in range(output_w) :
        im2col[i * output_w + j] = input_data[i * stride : i*stride + filter_h, j*stride : j*stride + filter_w].reshape(-1)

print(im2col)