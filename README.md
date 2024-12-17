# ResNet50 Skin Lesion CLassification

The project aims to develop a deep learning model for accurately classifying skin lesions into distinct categories, including melanoma, nevi, and benign lesions. Skin lesion classification is a critical task in dermatology, and automating this process using deep learning techniques can aid in early detection and improved patient outcomes.
Unlike standard datasets such as MNIST or CIFAR-10, which are limited in diversity, skin lesion classification presents challenges due to the variability and complexity of skin conditions.

The project utilizes the International Skin Imaging Collaboration (ISIC) dataset, which contains a diverse collection of dermatoscopic images encompassing various skin lesion types. The dataset is annotated with ground truth labels for lesion categories, providing a valuable resource for supervised learning tasks in skin lesion classification.

Dataset URL: https://challenge.isic-archive.com/data/

The version I have used it the 2016 version.


Task: Training a deep neural network to accurately classify skin lesions into predefined categories, including melanoma (malignant), nevi, and benign lesions. I will explore architectures such as ResNet, VGG,
or custom networks suitable for handling complex image features specific to dermatoscopic images. 
Data augmentation techniques such as rotation, flipping, scaling, and color jittering will be employed to improve model
generalization and robustness. The problem is a 2 class classification task given the labels of benign and malignant in the ground-truth files.


Relevant Papers:
1. Kassem, Mohamed A., et al.
"Machine learning and deep learning methods for skin lesion classification and
diagnosis: a systematic review.
" Diagnostics 11.8 (2021): 1390.
2. Lopez, Adria Romero, et al.
"Skin lesion classification from dermoscopic images using deep learning
techniques.
" 2017 13th IASTED international conference on biomedical engineering (BioMed). IEEE, 2017.
3. Benyahia, Samia, Boudjelal Meftah, and Olivier Lézoray.
lesion classification.
" Tissue and Cell 74 (2022): 101701.


To do this task I am going to be using 2 models, ResNet50 and VGG19.

### Why Use ResNet50?

**Deep Architecture with Skip Connections**:

ResNet50 (Residual Network with 50 layers) is known for its deep architecture, consisting of 50 layers. It employs residual (skip) connections, which help mitigate the vanishing gradient problem that can occur in very deep networks. This allows for training much deeper networks compared to traditional architectures.

Efficient Feature Learning:

The skip connections make it easier for the network to learn complex features by allowing gradients to flow directly through these connections during backpropagation. This improves performance without the risk of degradation seen in deep networks.

High Performance in Image Classification:

ResNet50 has demonstrated outstanding performance on image classification benchmarks like ImageNet, making it a reliable choice for medical image classification tasks where detailed feature extraction is crucial.

Transfer Learning Capabilities:

ResNet50 is often pre-trained on large datasets like ImageNet. By leveraging transfer learning, we can adapt the network to our skin lesion classification task without the need for extensive training data.
Architecture Characteristics

50 Layers:
Includes a combination of convolutional layers, batch normalization, ReLU activations, and pooling layers, structured into multiple residual blocks.

Residual Blocks:
These blocks consist of identity mappings that bypass one or more layers, ensuring the network can learn identity functions easily, which helps with deeper learning.
Pros:
- Handles deep architectures effectively.
Reduces the vanishing gradient problem.
- High accuracy and efficient training with transfer learning.
Cons:
- More computationally intensive compared to shallower networks.




### Why Use VGG19?

**Simple and Consistent Architecture:**

VGG19 (Visual Geometry Group network with 19 layers) is known for its simplicity and consistent structure. It uses very small 3x3 convolutional filters and a deep stack of convolutional layers to extract features.
Deep but Manageable Depth:

With 19 layers, VGG19 is deep enough to learn complex features but simpler than ResNet architectures. It doesn't use residual connections, relying purely on stacked convolutional layers.

Strong Baseline for Transfer Learning:

VGG19 has been widely used in transfer learning scenarios due to its success on the ImageNet dataset. Its architecture works well as a feature extractor, making it suitable for tasks where pre-trained models can be fine-tuned for medical images.

Good for Smaller Datasets:

Because of its simpler structure, VGG19 can perform well on smaller datasets where deeper architectures might overfit.

Architecture Characteristics

19 Layers:

Consists of 16 convolutional layers and 3 fully connected layers. Each convolutional block uses ReLU activations and max pooling layers for downsampling.

Uniform Convolutional Filters:

All convolutional layers use 3x3 filters, making the architecture straightforward and easy to understand.

Pros:

- Simple and intuitive architecture.

- Effective for transfer learning.

- Good for smaller datasets and less computationally intensive compared to ResNet50.

Cons:

- Prone to vanishing gradient issues for deeper networks.

- More computationally expensive than some modern architectures due to the large number of parameters.
