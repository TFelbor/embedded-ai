# Embedded Artificial Intelligence Portfolio

This repository contains my coursework, lab implementations, and research presentations for the **Embedded Artificial Intelligence** course at Polytech Nice Sophia. The projects focus on deploying Deep Learning models (CNNs) onto resource-constrained microcontrollers (STM32L4) and exploring unsupervised learning algorithms.

## 📂 Repository Structure

### 🧪 Labs: Edge AI Pipeline
A progressive series of labs demonstrating the full lifecycle of Embedded AI, from training to deployment on an **STM32L476RGT6** (Cortex-M4).

* **Lab 1: Environment Setup**
    * Established a Docker-based toolchain with TensorFlow and Arduino IDE.
    * Verified hardware constraints (80 MHz, 128 KB SRAM, 1 MB Flash).
    
* **Lab 2: CNNs & MicroAI Framework**
    * Designed and trained Convolutional Neural Networks (CNNs) on standard datasets (MNIST, UCI HAR).
    * **Key Achievement:** Manually implemented Dense and Convolutional layers in C to understand the low-level arithmetic of inference.
    * Used the **MicroAI** tool to automatically generate optimized, fixed-point C code from Keras models.

* **Lab 3: Human Activity Recognition (PolyHAR)**
    * **Goal:** Real-time Human Activity Recognition (HAR) on-device.
    * **Data:** Collected a custom accelerometer dataset (Positive/Negative activity classes) using the RFThings board.
    * **Modeling:** Trained a 1D-CNN in TensorFlow/Keras on the custom time-series data.
    * **Deployment:** Converted the model to a 16-bit fixed-point C library and integrated it into the microcontroller firmware for real-time inference (LED actuation upon detection).

### 🧠 Research Presentation: Growing Neural Gas (GNG)
Located in `/GNG_presentation`

An exploration of **Growing Neural Gas**, an unsupervised learning algorithm that learns the topology of data inputs without a predefined structure (unlike SOMs).

* **`GNG.py`**: A Python implementation of the GNG algorithm from scratch.
    * Includes a real-time visualization of the network "growing" to fit a non-linear dataset (Two Moons).
* **Slides**: `Self-Growing Neural Networks.pdf` - A detailed presentation on the algorithm's theory, math, and comparison to Self-Organizing Maps (SOM).

## 🛠️ Tech Stack & Skills
* **Hardware:** STM32L4 Microcontrollers, RFThings-AI Dev Kit, IMU Sensors.
* **Languages:** C/C++ (Firmware), Python (Training & Simulation).
* **Libraries:** TensorFlow/Keras, NumPy, MicroAI (C inference engine).
* **Tools:** Docker, Arduino IDE, Jupyter Notebooks.

## 🚀 Usage

**Running the GNG Simulation:**
```bash
cd GNG_presentation
python GNG.py
```
(Ensure you have numpy, matplotlib, and scikit-learn installed)
