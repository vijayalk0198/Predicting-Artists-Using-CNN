# Decoding Art using Machine Learning: A CNN-based Approach to Predict Artists from Artworks

## Introduction
This project explores the use of deep learning for artistic style recognition and artist identification. We implement convolutional neural networks (CNNs) to classify artworks based on their respective creators. Using techniques such as transfer learning, fine-tuning, and data augmentation, we aim to enhance the accuracy of predictions. The study focuses on understanding global and local artistic traits, leveraging residual networks like ResNet50 and ConvNeXt to optimize performance.

# Tools & Concepts
## Libraries Used:
- TensorFlow/Keras - Deep learning framework  
- OpenCV - Image processing  
- Matplotlib & Seaborn - Data visualization  
- Pandas & NumPy - Data handling  
- Scikit-learn - Performance evaluation  

## Concepts:
- Convolutional Neural Networks (CNNs)  
- Transfer Learning  
- Data Augmentation (Rotation, Flipping, Zooming, Translation)  
- Model Optimization (Fine-tuning, Regularization)  
- Evaluation Metrics (Accuracy, Precision, F1-score)  

# Methodology
1. **Dataset Selection**  
   - The "Best Artworks of All Time" dataset from Kaggle was used, containing images of paintings from 50 renowned artists.  

2. **Data Preprocessing**  
   - Images were resized to 224x224 pixels, normalized, and augmented to improve generalization.  

3. **Model Architectures**  
   - **Custom CNN Model**: Built from scratch with multiple convolutional and pooling layers.  
   - **Pre-trained Models**:  
     - VGG16  
     - ResNet50  
     - ConvNeXt  
   - Fine-tuned for our classification task.  

4. **Training & Validation**  
   - Models were trained using categorical cross-entropy loss and evaluated using accuracy and F1-score.  

5. **Hyperparameter Tuning**  
   - Batch size, dropout rates, number of layers, and regularization techniques were experimented with to enhance model performance.

# Results & Conclusion
- The baseline CNN model achieved ~36% accuracy, indicating the need for deeper architectures.  
- VGG16 and ResNet50 improved performance, but overfitting was observed.  
- ConvNeXt outperformed other models, achieving 74.4% test accuracy, demonstrating the advantage of modern convolutional architectures.  
- Data augmentation and fine-tuning played a crucial role in improving generalization.  

This project highlights the potential of deep learning for automated artist classification.  
