# Multibiometric Facial Recognition using ML and Deep Learning

This project implements a facial recognition system capable of working under various occlusions and across multiple modalities, including visible, thermal, and infrared images. It uses both traditional machine learning and deep learning models for detection and recognition tasks.

## 📚 Project Overview

- Built a multibiometric dataset of 22 subjects with 16 facial occlusion types (mask, scarf, goggles, cap, etc.)
- Collected over 1800 images across three modalities: **Visible**, **Thermal**, and **Infrared**
- Captured 5 head poses per combination: front, up, down, left, right

## 🧠 Implemented Models

### 🔍 Face Detection
- **Haar Cascade** (OpenCV)
- **HOG + SVM** (Dlib)
- **MTCNN** (Multitask CNN)

### 🧾 Face Recognition
- **FaceNet** (InceptionResnetV1-based embeddings)
- **ArcFace** (InsightFace-based embeddings)

## 📊 Evaluation

- Detection and recognition accuracies evaluated across:
  - Modalities: Visible, Thermal, Infrared
  - Difficulty levels: Normal, Easy, Medium, Hard (based on occlusion types)
- Cosine similarity used for embedding matching in recognition tasks
- Recognition tested using query–gallery image pairs

## 🗂 Dataset Format

Each image is named as: `id_addon_modality_pose.jpg`, where:
- `id`: subject number (01–22)
- `addon`: face occlusion (e.g., ma = mask, ca = cap, ga = goggles)
- `modality`: `v` = visible, `t` = thermal, `i` = infrared
- `pose`: `f` = front, `u` = up, `d` = down, `l` = left, `r` = right



