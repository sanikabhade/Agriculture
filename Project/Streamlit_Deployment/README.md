# Maharashtra Agricultural Market Price Forecasting

An interactive Streamlit dashboard for exploring historical agricultural market prices across Maharashtra and viewing precomputed crop-market price forecasts.

## Features
- Historical agricultural price analysis
- Commodity and market-wise price exploration
- Market comparison
- Interactive charts and summary statistics
- Forecast exploration for selected commodities and markets
- CSV download of forecast results
- Model information and evaluation metrics

## Dataset
The project uses a Maharashtra agricultural market dataset covering January 2024 to July 2025.
The dataset includes commodity, market, district, date, arrival quantity, modal price, weather, and soil-related features.
The LSTM forecasting dataset uses observations measured in Rs./Quintal.

## Methodology
The project uses a Long Short-Term Memory (LSTM) neural network to model historical agricultural modal prices.
The dashboard displays precomputed forecast results. It does not train the model or generate new predictions when the app runs.

### Model Evaluation Metrics
- R2 Score: 0.8175
- Mean Absolute Error (MAE): Rs. 531.13
- Root Mean Squared Error (RMSE): Rs. 869.27

## Forecast Limitations
The displayed 12-month forecasts are experimental projections based on a recursive rollout and assumptions about future input features.
The model was trained to predict the next available observation using the previous 14 observations. This does not establish validated accuracy for monthly forecasts over a 12-month horizon.
Future weather and arrival-related inputs are approximated using historical information. Forecasts should therefore be treated as indicative estimates, not guaranteed future market prices.

## Project Structure
```text
Agri_Project_Final_Deployment/
|-- app.py
|-- requirements.txt
|-- README.md
|-- data/
|   `-- Maharashtra_Agri_Master_2024_Jul2025.csv.gz
|-- outputs/
|   `-- Maharashtra_Crop_Market_12_Month_Forecast.csv
|-- model/
|-- notebooks/
```

## Run Locally
1. Install Python.
2. Clone or download this repository.
3. Open a terminal in the project directory.
4. Install dependencies:
```bash
pip install -r requirements.txt
```
5. Start the application:
```bash
streamlit run app.py
```

## Technologies Used
- Python
- Pandas
- NumPy
- TensorFlow / Keras (model development)
- Streamlit
- Plotly
- LSTM Neural Networks

## Disclaimer
This project is developed for academic and educational purposes. Forecasts are experimental and should not be used as the sole basis for agricultural trading, investment, or financial decisions.