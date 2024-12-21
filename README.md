# Insulator_fault_detector

This project provides a pipeline for detecting faults in electrical insulators using a YOLOv7-CBAM based object detection framework. It includes components for dataset preparation, training, and evaluation.

Directory Structure

GA-Kmeans

Contains scripts for optimizing YOLO anchor boxes using Genetic Algorithm (GA) and K-means clustering.
	•	GA-Kmeans.py: Implements GA and K-means for anchor box optimization.
	•	utils.py: Utility functions for the GA-Kmeans implementation.
	•	yolo_anchors.txt: Predefined anchor boxes for the YOLO model.

model_data

Stores model-related data and class information.
	•	class_names.txt: Defines the classes for insulator fault detection.

nets

Holds the neural network architecture and training scripts.
	•	__init__.py: Initializes the module.
	•	backbone.py: Backbone network implementation for YOLO.
	•	yolo.py: Defines the YOLO model architecture.
	•	yolo_training.py: Contains training logic and procedures.

utils

Utility scripts for data loading, bounding box manipulations, and training helpers.
	•	__init__.py: Initializes the module.
	•	callbacks.py: Defines callbacks for training.
	•	dataloader.py: Handles data loading and preprocessing.
	•	utils.py: General utility functions.
	•	utils_bbox.py: Functions for bounding box operations.
	•	utils_fit.py: Training utilities.
	•	utils_map.py: Computes mean Average Precision (mAP).

VOCdevkit

Contains scripts for dataset annotation, training, and prediction.
	•	predict.py: Script for making predictions using the trained YOLO model.
	•	README.md: Documentation for using VOCdevkit.
	•	train.py: Script for training the YOLO model.
	•	voc_annotation.py: Converts dataset annotations to YOLO format.
	•	yolo.py: A YOLO-related script specific to this dataset.
