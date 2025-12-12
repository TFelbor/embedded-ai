# Embedded AI Portfolio

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

<img width="325" height="285" alt="microcontroller" src="https://github.com/user-attachments/assets/3f3337d7-b78b-4f51-84de-58d8c9c278aa" />

*Figure 1: RFThings-AI Dev Kit board equipped with a STM32L476RGT6 Microcontroller*

<img width="1185" height="490" alt="plots" src="https://github.com/user-attachments/assets/9f16c64a-6863-482e-bc98-9a01463b8709" />

*Figure 2: Training Accuracy & Loss Curves*

### 🧠 Research Presentation: Growing Neural Gas (GNG)
Located in `/GNG_presentation`

An exploration of **Growing Neural Gas**, an unsupervised learning algorithm that learns the topology of data inputs without a predefined structure (unlike SOMs).

<img width="900" height="650" alt="vis1" src="https://github.com/user-attachments/assets/9db05e09-c156-4d37-92a5-251983f086e6" />

*Figure 3: Training -> Step 0 - 100*

<img width="900" height="650" alt="vis2" src="https://github.com/user-attachments/assets/8a0a1940-d13c-4c4d-a529-683179be36c8" />

*Figure 4: Training -> Step 100 - 1,000*

<img width="900" height="650" alt="vis3" src="https://github.com/user-attachments/assets/550dc022-644f-4ae4-83e8-dc82b39cf224" />

*Figure 5: Training -> Step 1,000 - 3,000*

<img width="900" height="650" alt="vis4" src="https://github.com/user-attachments/assets/743ff08a-dfa7-4529-a534-a740d99de2bd" />

*Figure 6: Training -> Step 3,000 - 8,000*

<img width="900" height="650" alt="vis5" src="https://github.com/user-attachments/assets/353f5648-3931-4251-8c69-1445b41252e2" />

*Figure 7: Training -> Step 8,000 - 12,000*

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
