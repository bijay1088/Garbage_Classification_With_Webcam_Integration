# Garbage Classification With Webcam Integration

This is a matlab project that used dataset from kaggle to train a model using AlexNet as Trasfer Learning. It was a university project where we had to train a model using transfer learning and get over 80% accuracy. While I was experimenting it, it took long time to train a model so I started saving the model after training. Then I thought why not use that model to classify the garbage from real life itself so I created a script that uses webcam and utilise that saved model to classify the garbage class in real time. 

**Dataset used in the project**: https://www.kaggle.com/datasets/mostafaabla/garbage-classification 

**Note**: 
Before training the model, I deleted some images to balance the dataset so that there is no bias after training. The original dataset had 15,150 images but I removed some to balance and my final images were 8406.
The data was also distributed with 75% in training, 15% for validation and rest of the 10% for testing.
![image](https://github.com/user-attachments/assets/96a0b4e4-0acf-471e-9b79-311e4794cfcd)




**Training Progress Chart**

![image](https://github.com/user-attachments/assets/820ec975-a000-49ea-9a89-be3d1b218e11)


**Test Accuracy**

![image](https://github.com/user-attachments/assets/eb9b7730-0d80-472e-8ae3-e90d2c64fb0d)


**Test using test images**

![image](https://github.com/user-attachments/assets/eb992956-2698-42b6-aeea-49c77be80e74)


**Confusion Matrix**

![image](https://github.com/user-attachments/assets/20861f4f-2344-4ea4-8485-630e925dedc7)


**Precision, Recall and F1 Score**

![image](https://github.com/user-attachments/assets/ac1abb6e-64c5-4758-93b0-97ca2bf060df)


**Some wrong predictions**

![image](https://github.com/user-attachments/assets/f1e97971-b06b-48e9-af8b-9feefde6e0f7)




