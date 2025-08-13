# Autonomous Fire Sensor and Human Detection System for First Responders  
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)  
[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)  
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)  

---

## Overview
This repository contains the design, implementation, and testing code for our **Senior Capstone Project (Jan–May 2025)**: an autonomous sensing platform that detects and locates people in smoke-filled environments to assist first responders during fires.

The system integrates **Texas Instruments mmWave cascading radar** with an **Intel RealSense depth camera** to combine RF point cloud data with optical depth imaging. A custom-trained machine learning model identifies human figures and reports their location and count in real time, even in low-visibility conditions.

---

## Features
- **Dual-Sensor Fusion**: Combines RF (mmWave radar) and optical (RealSense depth) data for enhanced detection accuracy.  
- **Autonomous Human Detection**: ML models trained to identify human figures through smoke and environmental noise.  
- **Real-Time Reporting**: Outputs human count and locations for rapid decision-making in emergencies.  
- **Custom Test Environment**: Fog simulation chamber to validate performance in realistic fire-like conditions.  
- **User Portal**: Simple interface for first responders to view detection data live.  

---

## System Architecture
```
[ TI mmWave Radar ] ---> RF Point Cloud ----┐
                                            │--> Sensor Fusion --> Human Detection (ML)
[ Intel RealSense ] --> Depth Images -------┘                         │
                                                                    Output to UI
```

---

## Technologies
**Hardware:** TI mmWave Cascading Imaging Radar, Intel RealSense Depth Camera  
**Software:** Python, OpenCV, ROS2, YOLO, NumPy, Intel RealSense SDK  
**ML Frameworks:** PyTorch / YOLOv8  
**Tools:** Docker, Git, Fog Simulation Test Rig  

---

## Repository Structure
```
📂 src/                # Core source code for data acquisition, fusion, and detection
📂 models/             # Trained machine learning models
📂 scripts/            # Utility scripts for data processing and testing
📂 docs/               # Documentation, diagrams, and system design files
📂 test/               # Test scripts and sample datasets
README.md              # Project documentation (this file)
LICENSE                # License information (MIT recommended)
```

---

## Getting Started

### Prerequisites
- Python 3.9+
- ROS2 Humble (or compatible version)
- Intel RealSense SDK
- mmWave Radar SDK (TI)
- PyTorch (with CUDA if using GPU)
- OpenCV

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR-USERNAME/fire-sensor-human-detection.git
cd fire-sensor-human-detection

# Install Python dependencies
pip install -r requirements.txt
```

---

## Usage
```bash
# Run sensor fusion and detection
python src/main.py
```
The system will:
1. Initialize radar and RealSense streams  
2. Perform sensor fusion  
3. Run ML-based human detection  
4. Output results to the terminal and optional UI  

---

## Testing
We used a **fog simulation chamber** to recreate low-visibility conditions.  
Test data is available in `test/` for reproducing results.

---

## Team
- Natalia Wilson
- Sameeha Boga  
- Daniel Fontaine  
- Arya Goyal  
- Eve Mooney  
- Daniela Salazar  
  

**Advisor:** Professor Jose Angel Martinez Lorenzo  

---

## License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
