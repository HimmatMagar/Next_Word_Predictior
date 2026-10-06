# Next Word Prediction

A web application that predicts the 5 most probable next words for a piece of text using an LSTM (Long Short-Term Memory) neural network. It has a responsive frontend and a FastAPI backend that returns predictions in real time.

## Features

- **LSTM-based predictions**: A trained deep learning model predicts the next word from the text entered so far.
- **Top-5 suggestions**: The 5 most probable next words are shown, ranked by confidence.
- **Real-time predictions**: Suggestions update as you type, using debounced requests to avoid flooding the server.
- **Clean UI**: Responsive layout with smooth animations.
- **Easy text editing**: Left-click a suggestion to insert it, right-click a word to remove it.
- **Error handling**: Clear error messages and graceful fallback when the server is unavailable.
- **CORS enabled**: Configured for cross-origin requests.

## Project Structure

```
Next_Word_Predictior/
├── app.py                          # FastAPI application entry point
├── main.py                         # Training script
├── requirements.txt                # Python dependencies
├── artificate/
│   ├── build_model/model.h5        # Trained LSTM model
│   ├── data_ingestion/quotes.csv   # Training data
│   └── data_transformation/        # Tokenizer and sequences
├── src/nextWordPrediction/pipeline/prediction_pipeline.py
├── templates/index.html            # Frontend UI
└── notebooks/                      # Jupyter notebooks for research
```

## Installation

### Prerequisites

- Python 3.12
- pip or conda

### Setup

1. Clone the repository:

   ```bash
   git clone <repository-url>
   cd Next_Word_Predictior
   ```

2. Create and activate a virtual environment:

   ```bash
   conda create -p env python==3.12 -y
   conda activate ./env
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Start the server

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` in your browser.

### How to use the app

1. Type text in the text area.
2. View the 5 most probable next words.
3. Left-click a suggestion to insert it.
4. Right-click a word to remove all occurrences of it.
5. Use the Clear and Example buttons for quick actions.

## API Reference

### GET /

Serves the frontend interface.

### POST /predict

Predicts the next word.

**Request**

```json
{ "word": "She walked into the" }
```

**Response**

```json
{
  "status": "success",
  "input": "She walked into the",
  "prediction": ["room", "house", "dark", "night", "garden"]
}
```

## Technical Details

### Backend

| Item | Value |
|---|---|
| Framework | FastAPI (async, CORS enabled) |
| Model | LSTM neural network |
| Sequence length | 252 tokens |
| Top-K | 5 predictions returned |

## Model Performance

The model is evaluated with cross-entropy loss and perplexity. Perplexity is the exponential of the loss:

```
perplexity = exp(loss)
```

Lower perplexity means the model is less uncertain about the next word. A perplexity of N means the model is, on average, as uncertain as if it were choosing uniformly among N words.

| Split | Loss | Perplexity |
|---|---|---|
| Testing | 7.21 | 1358 |
| Validation | 6.71 | 819 |


Notes:

- Perplexity is computed on the word-level vocabulary built from `quotes.csv`, so values are only comparable between runs that use the same tokenizer and vocabulary size.
- The training loss is averaged over each epoch while dropout is active, so it is normally higher than the validation loss, which is computed with dropout disabled.

## Development

### Train a new model

```bash
uv run main.py
```

### Explore the data

```bash
jupyter notebook notebooks/
```

## Troubleshooting

| Issue | Solution |
|---|---|
| Server unavailable | Make sure uvicorn is running on port 8000 |
| 405 Method Not Allowed | Restart the server and clear the browser cache |
| Empty predictions | Check that the input is not empty and that `tokenizer.pkl` exists |

## Contributing

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add feature"`
4. Push the branch: `git push origin feature/your-feature`
5. Open a pull request.

## License

MIT License. See the LICENSE file for details.

## Contact

For issues or suggestions, open an issue on GitHub.

---

Version: 1.0.0 | Status: Production Ready | Last Updated: March 2026
