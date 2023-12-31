__kernel void convolution(
    __global float *inp_data,
    __global float *oup_dat,
    __global float *fil_dat,
    int imageHeight,
    int imageWidth,
    int filterWidth,
    int halffilterSize)
{
    int gidX = get_global_id(0); // x軸方向(width)
    int gidY = get_global_id(1); // y軸方向(height)
    // 初始化輸出為 0
    float sum = 0.0f;
    // convolution
    for (int k = -halffilterSize; k <= halffilterSize; k++) {
        for (int l = -halffilterSize; l <= halffilterSize; l++) {
            // 計算輸入圖像的索引
            
            int y = gidY + k;
            int x = gidX + l;
            

            // 判斷是否需要 padding
            if (x < 0 || x >= imageWidth || y < 0 || y >=  imageHeight) {
                // padding 0
                continue;
            } else {
                // 計算 convolution
                sum += inp_data[y * imageWidth + x] * fil_dat[(k + halffilterSize) * filterWidth + l + halffilterSize];
            }
        }
    }

    // 將結果存儲到輸出圖像中
    oup_dat[gidY * imageWidth + gidX] = sum;
}