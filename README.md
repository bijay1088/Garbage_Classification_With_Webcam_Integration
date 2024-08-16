# Garbage Classification With Webcam Integration

This is a matlab project that used dataset from kaggle to train a model using AlexNet as Trasfer Learning. It was a university project where we had to train a model using transfer learning and get over 80% accuracy. While I was experimenting it, it took long time to train a model so I started saving the model after training. Then I thought why not use that model to classify the garbage from real life itself so I created a script that uses webcam and utilise that saved model to classify the garbage class in real time. 

**Dataset used in the project**: https://www.kaggle.com/datasets/mostafaabla/garbage-classification 

**Note**: 
Before training the model, I deleted some images to balance the dataset so that there is no bias after training. The original dataset had 15,150 images but I removed some to balance and my final images were 8406.
The data was also distributed with 75% in training, 15% for validation and rest of the 10% for testing.
![image](https://github.com/user-attachments/assets/73911128-51c7-4248-a8d9-9f35acb18f63)





**Training Progress Chart**

![Screenshot (58)](https://github.com/user-attachments/assets/ff49283b-547c-474d-b793-aa879abbbe6e)



**Test Accuracy**

![image](https://github.com/user-attachments/assets/eb9b7730-0d80-472e-8ae3-e90d2c64fb0d)


**Test using test images**

![image](https://github.com/user-attachments/assets/dc6f14b5-4234-4965-8cda-9e932319db68)


**Confusion Matrix**

![image](https://github.com/user-attachments/assets/504a78a9-b614-480a-94ac-5d1aa9025b51)


**Precision, Recall and F1 Score**

![image](https://github.com/user-attachments/assets/3ad361d2-f9dc-436c-9032-2ec798e9b138)


**Some wrong predictions**

![image](https://github.com/user-attachments/assets/c75061a9-d8c4-435d-a389-0ca7f6bd00dd)





