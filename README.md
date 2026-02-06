# 🛡️ Aegis Engine v2.8

**The Advanced Audio-to-MIDI Extraction Engine for Guitarists.**

Aegis Engine is a high-performance, AI-driven audio processing tool designed to convert messy guitar recordings into clean, professional-grade MIDI data and Tablature. Unlike traditional DSP-based converters, Aegis utilizes a hybrid architecture of **Computer Vision** and **Financial Trend Analysis** to isolate musical truth from physical noise.

---

## 🚀 Key Innovation: The Aegis Logic

### 1. Vision AI: Rake Pattern Detection
Most converters fail when a guitarist "rakes" across strings (fast broadband noise). Aegis treats the audio as a 2D image (Spectrogram) and uses pattern recognition to identify and isolate these vertical noise bursts before pitch tracking even begins.

### 2. Financial Algorithm: Trend Analysis
Inspired by stock market charting, Aegis applies a **Moving Average/Trend** logic to pitch data. It filters out high-frequency volatility (finger slides, fret buzz) and focuses on the stable "Trend" of the note, resulting in MIDI data that looks like a professionally written score.

### 3. Guardian Mode (Non-Destructive)
Aegis doesn't just delete data. Low-confidence segments are quarantined to a separate "SafeZone" MIDI track, allowing the user to recover nuanced performance data without cluttering the main "Aegis Prime" track.

---

## ✨ Features

- **Turbo Mode**: Multi-core parallel processing for 4x faster analysis.
- **Intelligent TAB Generator**: Automatically calculates the most ergonomic fingerings (string/fret) to minimize hand movement.
- **Selective Analysis**: High-precision time-range slicing—focus on that 10-second solo instead of the whole track.
- **Visual Feedback**: Real-time Spectrogram, Piano Roll, and Noise Mask reporting.
- **Dynamic Velocity**: RMSE-based energy mapping for realistic MIDI expression.

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/aegis-engine.git
cd aegis-engine

# Install dependencies
pip install -r requirements.txt

# Run the UI Center
streamlit run aegis_app.py
```

---

## 📱 Tech Stack

- **Core**: Python 3.9+
- **Analysis**: Librosa, NumPy, SciPy
- **MIDI**: Mido, PrettyMIDI
- **UI**: Streamlit
- **Visualization**: Matplotlib

---

## 📜 License
MIT License - Developed by **Aegis Senior Architecture**.
