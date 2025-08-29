# PM2.5 Prediction Using Machine Learning

This repository contains code and datasets for forecasting PM2.5 pollution levels in **Lucknow** using Artificial Neural Networks (ANNs).

## Project Overview
- Developed a machine learning model using **ANNs** to predict PM2.5 pollution levels.  
- Used **3+ years** of historical data from CPCB monitoring stations (synthetic data provided here).  
- Performed **feature engineering** using meteorological factors (temperature, humidity, wind speed).  
- Designed and trained a **multi-layer ANN model** with optimized hyperparameters for **time-series forecasting**.  

## Repository Structure
- `data/` → Contains the dataset (`pm25_data.csv`).  
- `src/` → Python scripts for preprocessing, training, and evaluation.  
- `notebooks/` → Jupyter notebooks for exploration and experiments.  
- `results/` → Model outputs, metrics, and plots.  
- `docs/` → Documentation and reports.  

## Installation
```bash
git clone <repo-url>
cd PM25_Prediction_Using_ML
pip install -r requirements.txt
```

## Usage
1. Preprocess the dataset:
```bash
python src/preprocess.py
```
2. Train the model:
```bash
python src/train.py
```
3. Evaluate the model:
```bash
python src/evaluate.py
```

## Results
- Achieved **R² = 0.87** with **low RMSE**.  
- Developed a smart **data-driven tool** for early warnings on pollution spikes.  

---
*Society of Civil Engineers, IIT Kanpur (Jun'24 - Jul'24)*
