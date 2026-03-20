# Next Word Prediction

A modern web application that predicts the next 5 most probable words using an LSTM (Long Short-Term Memory) neural network. The application features a clean, responsive UI and a FastAPI backend for seamless real-time predictions.

## Features

- **LSTM-Based Predictions**: Uses a trained deep learning model to predict the next word with high accuracy
- **Top-5 Suggestions**: Displays the 5 most probable next words ranked by confidence
- **Real-Time Predictions**: Instant feedback as you type with debounced requests
- **Clean UI**: Modern, responsive design with smooth animations
- **Easy Text Management**: Insert suggestions with a left-click, remove words with a right-click
- **Error Handling**: Graceful fallback with clear error messages
- **CORS-Enabled**: Fully configured for cross-origin requests

## Project Structure

```
Next_Word_Predictior/
├── app.py                          # FastAPI application entry point
├── main.py                         # Training script
├── requirements.txt                # Python dependencies
├── artificate/
│   ├── build_model/model.h5        # Pre-trained LSTM model
│   ├── data_ingestion/quotes.csv   # Training data
│   └── data_transformation/        # Tokenizer and sequences
├── src/nextWordPrediction/pipeline/prediction_pipeline.py
├── templates/index.html            # Frontend UI
└── notebooks/                      # Jupyter notebooks for research
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup Steps

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Next_Word_Predictior
   ```

2. Create and activate virtual environment:
   ```bash
   conda create -p env python==3.12 -y
   conda activate env/  # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the Server

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open browser: `http://127.0.0.1:8000`

### How to Use

1. Type text in the textarea
2. View the 5 most probable next words
3. **Left-click** to insert a suggestion
4. **Right-click** to remove all occurrences of a word
5. Use Clear/Example buttons for quick actions

## API Endpoints

### GET `/`
Serves the frontend interface.

### POST `/predict`
Predicts the next word.

**Request**:
```json
{ "word": "She walked into the" }
```

**Response**:
```json
{
  "status": "success",
  "input": "She walked into the",
  "prediction": ["room", "house", "dark", "night", "garden"]
}
```

## Technical Details

### Backend
- **Framework**: FastAPI (async, CORS-enabled)
- **Model**: LSTM Neural Network
- **Sequence Length**: 252 tokens
- **Top-K**: Returns top-5 predictions

### Frontend
- **Tech**: Vanilla HTML/CSS/JavaScript
- **Debounce**: 300ms to prevent server flooding
- **Responsive**: Mobile-friendly design

### Model Performance
- **Accuracy**: 48%
- **Loss**: 2.3266
- **Inference Time**: 50-100ms per prediction

## Development

### Train a New Model

```bash
python main.py
```

### Explore Data

```bash
jupyter notebook notebooks/
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Server unavailable | Ensure uvicorn is running on port 8000 |
| 405 Method Not Allowed | Restart server; clear browser cache |
| Empty predictions | Check input is not empty; verify tokenizer.pkl exists |

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -m 'Add feature'`
4. Push: `git push origin feature/your-feature`
5. Open Pull Request

## License

MIT License - See LICENSE file for details

## Contact

For issues or suggestions, open an issue on GitHub.

---

**Version**: 1.0.0 | **Status**: Production Ready | **Last Updated**: March 2026
