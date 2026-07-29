# Multi-Source RAG Research Assistant

A hybrid-search RAG system that answers questions from PDFs and web URLs with source citations.

## Features
- Upload PDFs and enter web URLs
- Hybrid search (semantic + keyword) for better accuracy
- Source tracing with confidence
- Export Q&A as text reports
- Built with LangChain, ChromaDB, Gemini

## How it works
1. Documents are chunked and embedded into a vector database
2. Questions are searched using both semantic + keyword retrieval
3. Retrieved chunks + question → Gemini → answer with citations

## Live Demo
[]

## Tech Stack
Python, LangChain, ChromaDB, Google Gemini, Streamlit