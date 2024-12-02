# RAG (Retrieval Augmented Generation)

Advanced document processing system combining PDF analysis with LLM for intelligent question answering.

## Features

- PDF text extraction and processing
- Semantic search with ChromaDB
- Context-aware responses using Mistral
- Streamlit web interface

## Installation

```bash
# Clone repository
git clone https://github.com/NassiraNguadii/RAG.git
cd RAG

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Install Ollama
curl https://ollama.ai/install.sh | sh
ollama pull mistral
```

## Usage

```bash
streamlit run src/app.py
```

## Project Structure

```
RAG/
├── Documentation/          # Project documentation
│   ├── Images/           # Documentation assets
│   └── Scripts/          # Documentation scripts
├── Notebooks/            # Development notebooks
├── conf/                 # Configuration files
├── src/                  # Source code
├── requirements.txt      # Dependencies
└── readthedocs.yml      # Documentation config
```

## Documentation

- Local: See `Documentation/` folder
- Online: [Read the Docs](https://rag-ia.readthedocs.io/en/latest/index.html/)

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Open Pull Request

## License

MIT License

## Contact

Questions and contributions welcome via GitHub Issues.