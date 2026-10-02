---
title: RetiScreen AI
emoji: 👁️
colorFrom: green
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# RetiScreen AI

This Hugging Face Docker Space serves the retinal-analysis website and its Python prediction API from the same app.

The selected model checkpoint and its routing/configuration files must be present in the Space repository at the paths used by the application. The Space starts `viva_server.py` on port 7860. Inference uses CUDA when the Space has GPU hardware enabled and falls back to CPU otherwise.

This research prototype is not a medical device and does not replace assessment by a qualified clinician.

## Streamlit Community Cloud

For Streamlit Community Cloud, set the app's main file path to `app.py`. Its `requirements.txt` installs CPU-only PyTorch, which is suitable for Streamlit Cloud's CPU runtime. The Dockerfile and `requirements_viva.txt` remain the Hugging Face Docker deployment configuration.
