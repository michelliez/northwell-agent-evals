"""Offline synthetic-query generation and the MiniLM/FAISS baseline.

Everything here is offline tooling for the future embeddings work and must
never enter the request path. The corpus schemas it operates on live in
``retrieval.chunk_models``; splitting and filtering live in ``evals``.
"""
