# 🌍 ClimaFact: A RAG-Powered Climate Policy Assistant

ClimaFact is an open-source retrieval-augmented generation (RAG) application that answers questions about climate risk, sustainability policy, and environmental planning by synthesizing information from authoritative sources like the EPA, IPCC, NOAA, and local government adaptation plans. Powered by state-of-the-art language models and vector search, ClimaFact helps policymakers, researchers, and the public get fast, grounded answers to complex climate questions.

> 🔍 Example Query: “What strategies has New York City adopted to mitigate coastal flooding by 2050?”

---

## 🚀 Features

- ✅ **RAG pipeline** using GPT-4 and embedded climate policy documents
- 📄 **Semantic search** over long, unstructured documents (PDFs, reports, legislation)
- 📚 **Citations + context snippets** returned for traceable answers
- 🔐 **Domain-aware prompts** and custom chunking to reduce hallucination
- 📊 **Evaluation-ready** with integration of RAGAS for answer quality assessment
- 🖥️ **Simple front-end** built with Streamlit for fast testing and demonstration

---

## 🛠️ Tech Stack

| Layer | Tool |
|-------|------|
| Language Model | `OpenAI GPT-4` or `Mistral` (via HuggingFace) |
| Vector Store | `FAISS` or `Chroma` |
| Embedding Model | `OpenAIEmbeddings` or `Instructor-XL` |
| Orchestration | `LangChain` or `LlamaIndex` |
| UI | `Streamlit` |
| Evaluation | `ragas`, `LlamaIndex eval` |

---

## 📂 Project Structure
Climafact/
│
├── Data_ingestion/
│ ├── scrape_epa.py
│ ├── parse_ipcc.py
│ └── Chunk_and_embed.py
│
├── rag_pipeline/
│ ├── retriever.py
│ ├── prompt_formatter.py
│ └── qa_generator.py
│
├── App/
│ └── App.py # Streamlit front end
│
├── eval/
│ └── evaluate_with_ragas.py
│
├── README.md
└── requirements.txt

---

## 🧪 Example Sources Used

- [EPA Climate Adaptation Plan](https://www.epa.gov/arc-x)
- [IPCC Sixth Assessment Report (AR6)](https://www.ipcc.ch/report/ar6/)
- [NOAA Sea Level Rise Technical Reports](https://coast.noaa.gov/slr/)
- [State of California Climate Action Plan](https://opr.ca.gov/planning/icarp/)

---

## 📈 Example Use Cases

- “Summarize the EPA’s position on methane regulation for oil and gas.”
- “What are the expected health risks of extreme heat in low-income neighborhoods?”
- “Compare the net-zero targets of New York and California.”
- “What funding is available for rural flood mitigation in the U.S.?”

---

## ✅ Running the App

1. **Install dependencies**

```bash
pip install -r requirements.txt

Embed documents
python data_ingestion/chunk_and_embed.py

Run the Streamlit app
streamlit run app/app.py
```

📊 Evaluation
We use ragas to evaluate:

Faithfulness

Groundedness

Relevance

Answer similarity to references (if available)

Optional: Add human-in-the-loop QA or custom eval set.

🧠 Future Work
Multi-agent reasoning over multiple sources

Local deployment using llama.cpp or vLLM

Domain-specific fine-tuning (e.g., climate-health)

Deployment to HuggingFace Spaces or Streamlit Cloud

Web form integration for policy planning tools

🧵 Author
Developed by Kevin Chan, a statistics & data science graduate passionate about building real-world GenAI systems for sustainability, public health, and good governance.

🌱 License
MIT License. Use freely, credit encouraged.
