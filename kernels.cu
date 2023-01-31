__global__ void mean_filter(float *img, float* res, int kernel_size, int h, int w){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if (x > kernel_size/2 && x < w - kernel_size/2 && 
    y > kernel_size/2 && y < h - kernel_size/2){
    
    float sum = 0.0;
    for (int hh=-(kernel_size/2); hh < kernel_size/2+1; hh++){
      for (int ww=-(kernel_size/2); ww < kernel_size/2+1; ww++){
        int idx = (y + hh) * w + x + ww;
        sum += img[idx];
      }
    }
      
    res[y*w + x] = sum/(kernel_size*kernel_size);
      
  }
}
__global__ void rgb2gray_gpu(float *img, float* res, int h, int w){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h){
    int t_idx = y*w + x;
  
    res[t_idx] = (img[t_idx*3] + img[t_idx*3+1] + img[t_idx*3+2])/3;
    
  }
}

__global__ void warpPerspectiveGpu(float *img, float* res, float* H, int h, int w, int c){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;


  if (x < w && y < h){
    float s_x = H[0]*x + H[1]*y + H[2];
    float s_y = H[3]*x + H[4]*y + H[5];
    float s_z = H[6]*x + H[7]*y + H[8];
    
    s_x /= s_z;
    s_y /= s_z;
    float s_x_f = floorf(s_x);
    float s_y_f = floorf(s_y);

    if (s_x_f < 0 || s_y_f < 0 || s_x_f + 1 >= w || s_y_f + 1 >= h) return;
    
    float w1 = (s_x - s_x_f)*(s_y - s_y_f);
    float w2 = (s_x - s_x_f)*(s_y_f + 1 - s_y);
    float w3 = (s_x_f + 1 - s_x) * (s_y - s_y_f);
    float w4 = (s_x_f + 1 - s_x) * (s_y_f + 1 - s_y);

    int idx1 = (int)(s_y_f+1) * w + (int)s_x_f+1;
    int idx2 = (int)s_y_f * w + (int)s_x_f+1;
    int idx3 = (int)(s_y_f+1) * w + (int)s_x_f;
    int idx4 = (int)(s_y_f) * w + (int)s_x_f; 

    int t_idx = y*w + x;

    for (int i = 0; i < c; i++){
      res[t_idx*c+i] = img[idx1*c+i]*w1 + img[idx2*c+i]*w2 +
                      img[idx3*c+i]*w3 + img[idx4*c+i]*w4;
      
    }

  }
}

__global__ void compute_disparity(float *img1, float *img2, int *disparity, int h, int w, int max_disp, int kernel_size){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < w && y < h){
    int disp = 0;
    float min_ssd = 1e7;
    
    for(int i = max_disp; i >= 0 && x - i >= 0; i--){
      float ssd = 0.0;
      float cnt = 0.0;
     
      for(int j = -(kernel_size/2); j < kernel_size/2+1; j++){
        for(int k = -(kernel_size/2); k < kernel_size/2+1; k++){
          int xx1 = x + j;
          int xx2 = x - i + j;
          int yy = y + k;
          if (xx1 < w && xx1 >= 0 && xx2 < w && xx2 >= 0 && yy < h && yy >= 0){
            cnt++;
            int idx1 = yy*w + xx1;
            int idx2 = yy*w + xx2;
            for(int c = 0; c < 3; c++){
              ssd += (img1[idx1*3+c] - img2[idx2*3+c])*(img1[idx1*3+c] - img2[idx2*3+c]);
            }
          }
        }
      }
      if (cnt > 0) ssd /= cnt;
      if (ssd < min_ssd){
        disp = i;
        min_ssd = ssd;
      }
    }
    disparity[y*w + x] = disp;
  }
}


__global__ void gaussian_smoothing(float *img, float *res, float* gauss_kernel, int ks, int h, int w){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if ( x > ks/2 && x < w - ks/2 && y > ks/2 && y < h - ks/2){
    for(int c = 0; c < 3; c++){
      float sum = 0.0;
      for(int xx = 0; xx < ks; xx++){
        for(int yy = 0; yy < ks; yy++){
          int idx_img = (y + yy - ks/2) * w + (x + xx - ks/2);
          int idx_k = yy * ks + xx;
          sum += img[idx_img*3+c] * gauss_kernel[idx_k];
        }
      }
      int idx = y*w + x;
      res[idx*3 + c] = sum; 
    }
  }
}
