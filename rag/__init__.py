"""ClearTrace Module 3 — Attribution Engine + RAG AI Chatbot.

This module provides:
  1. Source attribution — identifies which pollution sources (traffic, industry,
     construction, waste, power) likely contribute to AQI at a given location.
  2. RAG-powered chatbot — answers health and air-quality questions by combining
     CPCB guidelines (via FAISS) with live forecast data and crowd reports.

Tech stack: FastAPI · Groq (gemma2-9b-it) · FAISS · sentence-transformers
"""
