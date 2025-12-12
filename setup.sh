#!/bin/bash

echo "--- 1. Installing System Dependencies (FluidSynth) ---"
sudo apt-get update
sudo apt-get install -y fluidsynth fluid-soundfont-gm build-essential

echo "--- 2. Setting up SoundFont ---"
# Cerca il SoundFont installato dal sistema e lo copia nella cartella del progetto
# come 'soundfont.sf2' affinché app.py lo trovi.
if [ -f "/usr/share/sounds/sf2/FluidR3_GM.sf2" ]; then
    echo "Found FluidR3_GM. Copying to ./soundfont.sf2"
    cp /usr/share/sounds/sf2/FluidR3_GM.sf2 ./soundfont.sf2
else
    echo "System SoundFont not found. Downloading a fallback..."
    wget -qO soundfont.sf2 https://gitlab.com/musescore/MuseScore/-/raw/master/share/sound/FluidR3Mono_GM.sf3
fi

echo "--- 3. Installing Python Dependencies ---"
pip install torch \
            transformers \
            huggingface_hub \
            gradio \
            midi2audio \
            mido \
            pyfluidsynth \
            tqdm \
            sentence-transformers

echo "--- Setup Complete! You can now run 'python app.py' ---"
