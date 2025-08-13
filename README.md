# House Price Prediction

This project implements an end-to-end machine learning pipeline for predicting house prices using a Docker-based workflow. The pipeline includes data processing, model training, and MLflow experiment tracking.

## Project Structure
```
house_price_prediction/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # Utilities untuk memuat data
│   │   └── data_processor.py   # Pipeline pra-pemrosesan data
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py           # Definisi arsitektur model
│   │   └── trainer.py         # Logika pelatihan model
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py          # Konfigurasi logging
│   │   └── config.py          # Manajemen konfigurasi
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   └── test_model.py
├── notebooks/
│   └── exploration.ipynb
├── config/
│   └── config.yaml
├── requirements.txt
├── Dockerfile
└── README.md
```

## Features

- Pipeline pra-pemrosesan data dengan scikit-learn
- Beberapa model regresi (Linear Regression, Random Forest Regressor, Gradient Boosting Regressor)
- Pelacakan eksperimen dan model registry MLflow
- Eksekusi pipeline menggunakan Docker
- Sistem logging yang komprehensif
- Manajemen konfigurasi
- Struktur kode yang siap produksi

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/house_price_prediction.git
cd house_price_prediction
```

2. Pastikan Docker sudah terinstal dan berjalan di sistem Anda.

# Data Configuration
data_path: "data/train.csv"
target_column: "SalePrice"
preprocessing_path: "models/preprocessing"

# MLflow Configuration
mlflow:
  tracking_uri: "sqlite:///mlflow.db"
  experiment_name: "house_price_prediction"

# Model Parameters
model_params:
  random_forest:
    n_estimators: 100
    max_depth: 10
    random_state: 42


## Usage
  Letakkan file data Anda (train.csv) di direktori data:
- cp path/to/your/train.csv data/

  Bangun (build) Docker image dari Dockerfile yang ada di root direktori proyek.
- docker build -t house-price-pipeline .

  Jalankan pipeline pelatihan di dalam container Docker. Ini akan menjalankan
- docker run --rm -v $(pwd):/app house-price-pipeline

  Setelah pipeline selesai berjalan, file mlflow.db dan artefak model akan tersimpan di direktori lokal Anda.

  Lihat eksperimen di MLflow:
  mlflow ui --backend-store-uri sqlite:///mlflow.db

## Model Training
Pipeline ini melatih tiga jenis model:
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

Model terbaik dipilih berdasarkan nilai RMSE terendah dan secara otomatis didaftarkan di MLflow untuk penggunaan produksi.

## Monitoring
- Log disimpan di direktori logs/
- Informasi pelacakan MLflow disimpan di mlflow.db
- Artefak model disimpan di models/

## Development
  Running Tests
  pytest tests/

## Adding New Models
- Perbarui src/models/model.py dengan konfigurasi model baru Anda
- Tambahkan parameter spesifik model di config.yaml
- Perbarui pipeline pelatihan jika diperlukan

Co## ntributing
- Fork the repository
- Buat feature branch
- Commit perubahan Anda
- Push ke branch
- Buat Pull Request
## License

This project is licensed under the MIT License - see the LICENSE file for details.
