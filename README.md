# TTS Studio
**TTS Studio** is a professional-grade, GPU-accelerated storyboard interface for high-fidelity Text-to-Speech generation. Built for creators who need local privacy and studio-quality voice cloning without the subscription fees.

<img width="1363" height="847" alt="Demo" src="https://github.com/user-attachments/assets/19e4f05b-05cc-4ba7-b95f-d2fae862b633" />

### ✨ Key Features
• **Advanced Voice Cloning**: Clone any voice using just a 6-30 second reference sample. The app captures timbre and emotion with local **XTTSv2** technology.</br>
• **Storyboard Workflow**: Don't just generate text; build a project. Use Text Blocks to organize different speakers, languages, and settings in one timeline.</br>
• **Local AI Assistant**: Powered by **Qwen 2.5 (3B)**, your built-in companion can **rewrite, translate, or expand** your script directly inside the app.</br>
• **Multilingual Mastery**: Support for over 15 languages including **English, Russian, German, Spanish, and Japanese** using high-speed **MMS (Meta)** and **XTTSv2 models**.</br>
• **Real-time Waveforms**: Visualize your audio as it's generated with interactive seeking and volume controls.

### 💻 System Requirements
To ensure smooth generation, we recommend the following:</br>
• **GPU**: NVIDIA RTX Series (4GB+ VRAM recommended).</br>
• **RAM**: 8GB Minimum.</br>
• **Disk Space**: 5GB (to accommodate the local AI models).</br>
• **OS**: Windows 10/11 (64-bit).

### ⚠️ Required tools/libs for running the source code
• Microsoft C++ Build Tools (MSVC) **(latest)**</br>
• Python 3.11</br>
• **Python libs**: check the **"libs_versions.txt"** file for more information.

### 📜 Technical Credits
• **Engines**: [Coqui TTS (XTTSv2)](https://huggingface.co/coqui/XTTS-v2), [Meta MMS](https://huggingface.co/facebook/mms-tts-ron), [Alibaba Qwen](https://huggingface.co/Qwen/Qwen2.5-3B).</br>
• **UI Framework**: CustomTkinter.</br>
• **Audio Engine**: Pygame & Scipy.

### ⚖️ Usage & Licensing
TTS Studio is released under a Non-Commercial License.</br>
• The **XTTSv2** engine is governed by the [Coqui CPML](https://huggingface.co/coqui/XTTS-v2/blob/main/LICENSE.txt), which prohibits commercial use.</br>
• The **MMS** models are released under [CC-BY-NC 4.0.](https://creativecommons.org/licenses/by-nc/4.0/).</br>
• The **Qwen2.5-3B** model is released under the [Qwen Research](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE) license.</br>
• ⚠️**Commercial Use**: Any audio generated with this tool is for **personal projects only**. You are legally prohibited from using the output for monetized content or paid services.
