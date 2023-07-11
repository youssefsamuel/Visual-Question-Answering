# Visual-Question-Answering
A model that takes as input a question and an image, pass them into CLIP model for encoding, and then predict the answer to the question, associated with the answer confidence and answer type.
#	Introduction
##	Problem statement
Visual Question Answering (VQA) is an emerging field in computer vision and natural language processing that focuses on answering open-ended questions based on images. This task has numerous applications, including medical VQA, education, surveillance, and more. In this assignment, we will explore the VizWiz dataset, specifically designed to train models to assist visually impaired individuals.

The VizWiz dataset is a comprehensive VQA dataset comprising 20,500 image-question pairs. Each image is associated with a corresponding question, and for each question, there are 10 crowdsourced answers. This dataset plays a crucial role in developing models that can effectively answer questions based on visual information, aiding visually impaired individuals in accessing visual content.
##	Dataset Description
The VizWiz dataset is a Visual Question Answering (VQA) dataset specifically designed to assist visually impaired individuals. It consists of 20,500 image-question pairs, where each image is associated with a corresponding question, and each question has 10 crowdsourced answers. This dataset aims to facilitate the development of models that can effectively answer questions based on visual content, thereby enabling visually impaired individuals to access and comprehend visual information.
The dataset is divided into training and validation sets, with the following distribution:
1. 	Training Set:
a. 	20,523 image/question pairs
b. 	205,230 answer/answer confidence pairs
2. 	Validation Set:
a. 	4,319 image/question pairs
b. 	43,190 answer/answer confidence pairs

#	Classes
##	Dataset Class
Purpose: The main purpose of the myDataset class is to create a PyTorch dataset object that can be used for training, validation, or testing. It encapsulates the VizWiz dataset, providing an interface to access image embeddings, answers, answer types, and answerability information for a given index.

### Methods
1. 	get_item Method: The getitem method is responsible for returning a specific item from the dataset, given an index.
2. 	len Method:	The len method returns the total number of image-question pairs in the dataset. It is used to determine the length of the dataset.

##	Model Class
###	VQA Model
Purpose: The VQAModel class represents the architecture of the Visual Question Answering (VQA) model. It is implemented as a PyTorch module and consists of multiple layers and operations to process input features and generate predictions.
###	Answerability Model
The AnswerabilityModel class represents a PyTorch module for answerability prediction. Here's a brief explanation of its purpose and implementation:

#	Execution Steps
##	Loading CLIP model
CLIP Model: The CLIP (Contrastive Language-Image Pretraining) model is a powerful deep learning model developed by OpenAI. It is designed to understand and bridge the gap between natural language and visual content. CLIP leverages large-scale pretraining on a vast corpus of text and images, enabling it to learn joint representations of both modalities. By learning to associate images and their corresponding textual descriptions, CLIP becomes capable of performing various vision and language tasks, including image classification, object detection, and visual question answering.

Importance of CLIP in VQA: In the context of Visual Question Answering (VQA), the CLIP model is utilized as both the image encoder and text encoder. The pre-trained CLIP model provides rich and meaningful representations for both images and textual questions. By utilizing the CLIP model, the VQA model can leverage the comprehensive understanding of visual and textual information encoded in CLIP's pre-trained weights, thus enhancing its performance in answering questions based on visual content.
##	Extracting data from annotations
This part is used to extract data from the given files. The data needed to be extracted: the answers for each question, the answer types, and the answerability of the question.
Also, the most important step in the part is to get the most common answer to each question, to be used in training.

##	One hot encoding for labels
The purpose of this code segment is to transform the answer labels and answer types into binary representations using one-hot encoding. This conversion allows the VQA model to handle categorical data by representing them as binary vectors, which can be used as input for training the model. Additionally, the data is split into training and validation sets for further processing and evaluation.

##	Encoding of data using CLIP
The purpose of this code segment is to extract visual and textual features from the training and the validations images and questions using the CLIP model. 

##	Splitting
We need to split the training set into train and test. 

##	Loading the extracted features into a file
We needed to load the extracted features from clip model into a file to save some memory space. Then, the dataset class will need to get the path of these files to be able to return the features of a given element.

##	Creating instances of datasets and dataloader
In this step we created object of the class dataset for the training, testing and validation set, and we pass to the constructor all the data needed for the dataset class to get an item from the set.

The DataLoader is a PyTorch utility that helps in efficiently loading and processing data during model training or evaluation. In this code snippet, a DataLoader named val_dataloader is created for the validation dataset (valDataset).
The purpose of the DataLoader is to provide an iterable over the validation dataset, allowing you to easily access batches of data during the validation process. 

##	Training and running model
The run_model function appears to be a utility function for training and evaluating a given model using a specified data loader (dataloader). 

## Computing Accuracy
At each epoch, we calculate the loss of the epoch and we get the accuracy. 
