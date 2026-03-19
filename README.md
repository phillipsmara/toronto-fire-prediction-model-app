# 🚒 Toronto Fire Response Time Predictor

A machine learning web app that predicts Toronto Fire Services emergency response times based on incident type, location, time of day, and other contextual factors.

**[Live App →](https://phillipsmara-toronto-fire-prediction-model-app-appapp-xgf7fm.streamlit.app/)**

---

## Overview

This project uses over a decade of Toronto Fire Services incident data (2012–2023) to train an XGBoost regression model that estimates how long it will take for fire services to arrive at an emergency. The app allows users to input incident details and receive an instant response time prediction along with a performance benchmark against Toronto's 4-minute target.

---

## Features 

- Predicts fire response time in minutes and seconds
- Benchmarks prediction against Toronto's 4-minute response target
- Trained on 10+ years of real Toronto Fire Services open data
- Spatial features including distance to nearest fire station and hydrant density
- Temporal features including hour of day, day of week, and month
- Covers 130+ incident types

---

## Data Sources

All data sourced from the [City of Toronto Open Data Portal](https://open.toronto.ca/):

| Dataset | Description |
|---|---|
| Fire Services Emergency Incident Basic Detail | Incident records 2012–2023 |
| Fire Station Locations | Coordinates of all Toronto fire stations |
| Fire Hydrant Locations | Locations of fire hydrants across the city |
| Toronto Fire Services Run Areas | Station service area boundaries |

---

## Model

- **Algorithm:** XGBoost Regression
- **Target:** Response time in minutes (alarm to arrival)
- **Features:** Incident type, call source, alarm level, ward, time features, distance to nearest station, hydrant density within 500m
- **Tuning:** 3-fold cross-validation grid search over learning rate, depth, and estimator count
- **Evaluation:** MAE, RMSE, R², and % predictions within ±1 and ±2 minutes

---

## Project Structure

```
toronto-fire-prediction-model-app/
├── app/
│   └── app.py                  
├── src/
│   ├── config.py               
│   ├── pipeline.py             
│   ├── data/
│   │   ├── api_client.py       
 downloader
│   │   ├── loader.py           
│   │   ├── preprocessing.py    
│   │   └── features.py         
│   └── modeling/
│       ├── train.py            
        ├── evaluation
│       ├── predict.py          
│       └── explainability.py   
importance plots
├── models/                     
├── data/
│   ├── raw/                    
(gitignored)
│   └── processed/             
(gitignored)
├── main.py                    
└── requirements.txt
```