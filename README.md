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
