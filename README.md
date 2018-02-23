# COMP6208-AutoEncoder

## Project Brief
For our Advanced Machine Learning Assignment, our team will be working on the topic of image compression using machine learning techniques. Traditional codecs are hard-coded solutions and we would like to explore machine learning's adaptive nature in its application to image compression. We decided to approach the problem through different, mostly deep learning based, techniques as discussed in research papers that showed promising results, sparking our interest. Two specific methods that we are exploring include traditional autoencoding, where one trains an encoder that maps an image to small vector and decoder that maps the vector back to the original image, and upscaling where one trains an algorithm to convert downscaled images to the original size ones with the hope of preserving as many details from the original image as possible.

Initially, two of us are going to be focused on autoencoding and two on upscaling. Namely: Ishaan will look into a traditional autoencoder approach based on CNN, GRUs and GANs [1]. Michael is going to explore more traditional methods, including a single bottleneck layer. Mateusz will look at upscaling images with CNNs [2], then proceeding onto autoencoders with CNN. Maciej will look into upscaling of images using a Generative Adversarial Network [3].
To evaluate the performance of the chosen techniques in a consistent manner, we plan to use the same standard image datasets: MNIST for small b&w preliminary testing, CIFAR for small colour images, and Open Image Dataset V3 (to test on bigger images). Additionally, we are aiming to assess machine learning's usefulness in the field by comparing it against a standard image compression programs such as JPEG, JPEG2000, and WebP.

[1] O. Rippel and L. Bourdev, “Real-Time Adaptive Image Compression,” ArXiv e-prints, 2017.

[2] Dong C et al. "Image super-resolution using deep convolutional networks." 2016.

[3] C. Ledig et al., “Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network,” 2016.
