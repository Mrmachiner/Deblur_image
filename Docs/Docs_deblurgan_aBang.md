# Documents Deblur-Gan 

## Generative Adversarial Networks (GAN)
### Tìm hiểu về [Deblur-Gan](https://www.sicara.ai/blog/2018-03-20-GAN-with-Keras-application-to-image-deblurring)
1. Khái niệm: GAN thuộc nhóm generative model. Với Generative nghĩ la có khả năng sinh dữ liệu , Network mạng (mô hình) và Adversarial đối nghịch (nghĩa là GAN sẽ có 2 mạng đối nghịch nhau).
2. Generator và Discriminator
   * Generator (G) sẽ cố gắng sinh ra các dữ liệu giống như thật. 
   * Discriminator (D) sẽ cố gắng phân biệt đâu là dữ liêu thật, đâu là dữ liệu giả do Gen sinh ra.
  ![GAN_Architecture](../deblur-gan-master/Docs/imgDocs/GAN_Architecture.png)
   * Việc G cứ tạo ra dữ liệu giả giống thật đưa cho D phân biệt rồi D chỉ lại cho G chỗ chưa được thật như vậy sẽ giúp 2 mạng tự dạy nhau học và cùng cải tiến nhau.
   * Mục tiêu ở đây là maximize D(x) (giá trị dự đoán với ảnh x) và minimize D(G(z)) (giá trị dự đoán với ảnh z được sinh ra từ G)
3. The Generator
   * Ứng dụng GAN vào khôi phục ảnh mờ thì ở đây việc của Generator sẽ là tái tạo lại ảnh sắc nét nhất có thể với đầu vào là một bức ảnh bị mờ.
   * Vì là mạng học sâu nên việc xảy ra vanishing/exploding gradient là khó có thể tránh khỏi. Để hạn chế việc đó thì ở đây ta có dùng ResNet blocks (9 block)
   * [Resnet](Docs/pdfDocs/Resnet.pdf).
     * Ý tưởng chung của Resnet là việc có thể skip qua 1 hay nhiều lớp, và việc tính đạo hàm cũng dễ dàng hơn do.
     * ![imgRes](../deblur-gan-master/Docs/imgDocs/Resnet.png)
   * Cấu trúc mạng DeblurGan Generator  ![im](../deblur-gan-master/Docs/imgDocs/ArchitectureGan.jpeg)
4. The Discriminator
   * Đối nghịch lại với Generator thì ở đây việc của Discriminator sẽ cố gắng phân biệt ảnh thật (ảnh gốc không bị mờ ) và ảnh giả (ảnh được khôi phục từ ảnh mờ) 
   * Discriminator Architecture

      |Name|Filter|Kernel|Stride|Input|OutPut|
      |-|-|-|-|-|-|
      |Conv2D|64|(4, 4)|2|(256, 256, 3)|(128, 128, 64)
      |LeakyReLU|
      |Conv2D|64|(4, 4)|2|(128, 128, 64)|(64, 64, 64)
      |BN+LeakyReLU|
      |Conv2D|128|(4, 4)|2|(64, 64, 64)|(32, 32, 128)
      |BN+LeakyReLU|
      |Conv2D|256|(4, 4)|2|(32, 32, 128)|(16, 16, 256)
      |BN+LeakyReLU|
      |Conv2D|512|(4, 4)|1|(16, 16, 256)|(16, 16, 512)
      |BN+LeakyReLU|
      |Conv2D|1|(4, 4)|1|(16, 16, 256)|(16, 16, 1)
      |Flatten|
      |Dense(1024, 'tanh')|
      |Dense(1, 'sigmoid')|
5. Hàm Loss và Optimizer
   * Loss Function ![img](../deblur-gan-master/Docs/imgDocs/Lossfunction.png)
     * [L1 (MAE) ](https://en.wikipedia.org/wiki/Mean_absolute_error)
     * Perceptual Loss ?
     *  ![img](../deblur-gan-master/Docs/imgDocs/Lx.png)
        ![Perceptual_loss](../deblur-gan-master/Docs/imgDocs/perceptual_loss.png)
     * [Wasserstein Loss](https://arxiv.org/pdf/1701.07875.pdf) 
     * ![img](../deblur-gan-master/Docs/imgDocs/Lgan.png)
        ![img_Wloss](../deblur-gan-master/Docs/imgDocs/wasserstein_loss.png)
   * Optimizer [Adam](https://keras.io/optimizers/#adam)
6. Dataset
   * Sharp image 
     * [GoProAll](https://drive.google.com/file/d/1SlURvdQsokgsoyTosAaELc4zRjQz9T2U/view)
     * [Kadid](http://database.mmsp-kn.de/kadid-10k-database.html)
     * [GoPro DataSet](https://drive.google.com/file/d/1H0PIXvJH4c40pk7ou6nAwoxuR4Qh_Sa2/view)
   * Blur image
     * Convolution with kernel same same 
     * ![Kerne](../deblur-gan-master/Docs/imgDocs/kernel_blur.png)
7. Statistics
   
   * Result 
   * Size (256x256) ![Img](../deblur-gan-master/Docs/imgDocs/results0.png)
   * Size (1280x720) ![Img](../deblur-gan-master/Docs/imgDocs/Sharp.png) ![Img](../deblur-gan-master/Docs/imgDocs/Blur.png) ![Img](../deblur-gan-master/Docs/imgDocs/Deblur.png)


   |STT|Epoch|Name Model Gen|Sample Train|Bath Size|Critic Updates|File Sample Test|File Sample Out|Average Score Sharp |Average Score Deblur|Ratio|
   |-|-|-|-|-|-|-|-|-|-|-|
   |1|50|[generator_49_298.h5](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/Weight/Epoch50/)|2100|2|5|[100](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/blur)|[(256x256)](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/GAN_gen49_epoch50_sharp/256x256_out)|3162.7987|1816.9127|57.4463%
   |2|100|[generator_99_243.h5](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/Weight/Epoch100/210/)|2100|2|5|[100](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/blur)|[(256x256)](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/GAN_gen99_epoch100_sharp/256x256_out)|3162.7987|1849.6389|58.4811%
   |3|?|[generator.h5](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/scripts)|?|?|?|[100](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/blur)|[(256x256)](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/gan_real5_sharp/256x256_out)|3162.7987|1965.7959|62.1536%
   |4|50|[generator_49_298.h5](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/Weight/Epoch50/)|2100|2|5|[100](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/blur)|[(1280x720)](/home/minhhoang/Desktop/score_Img_valapcian/Folder_calculate_score/Gen49/img_deblur)|196.3081|208.8575|106.3927%
   |5|100|[generator_99_243.h5](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/Weight/Epoch100/210/)|2100|2|5|[100](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/blur)|[(1280x720)](/home/minhhoang/Desktop/score_Img_valapcian/Folder_calculate_score/Gen99/img_deblur)|196.3081|224.3599|114.2897%
   |6|?|[generator.h5](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/scripts)|?|?|?|[100](/home/minhhoang/Desktop/MinhHoang/ML_DL_inter/deblur-gan-master/GOPRO_Large/test/GOPR0384_11_05/blur)|[(1280x720)](/home/minhhoang/Desktop/score_Img_valapcian/Folder_calculate_score/Gen5_real/img_deblur)|196.3081|28267.7345|143.9968%