"""
Generate waveform and spectrogram visualizations for audio samples.
Uses scipy (no librosa required).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from pathlib import Path

# Paths
AUDIO_DIR = Path("../../slides/audio/week05")
OUTPUT_DIR = Path("../../slides/images/week05")

def visualize_audio_sample(audio_path, title, output_name):
    """Create waveform + spectrogram visualization."""
    sr, y = wavfile.read(audio_path)
    if y.dtype == np.int16:
        y = y.astype(float) / 32768.0
    if len(y.shape) > 1:
        y = y.mean(axis=1)  # Convert to mono

    duration = len(y) / sr
    times = np.arange(len(y)) / sr

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Waveform
    axes[0].plot(times, y, color='steelblue', linewidth=0.5)
    axes[0].set_xlabel('Time (seconds)', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Waveform', fontsize=12, fontweight='bold')
    axes[0].set_xlim(0, duration)
    axes[0].grid(True, alpha=0.3)

    # Spectrogram
    f, t, Sxx = signal.spectrogram(y, sr, nperseg=256, noverlap=200)
    axes[1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
    axes[1].set_xlabel('Time (seconds)', fontsize=11)
    axes[1].set_ylabel('Frequency (Hz)', fontsize=11)
    axes[1].set_title('Spectrogram', fontsize=12, fontweight='bold')
    axes[1].set_ylim(0, min(8000, sr/2))

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / output_name, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_name}")

def create_spectrogram_explainer():
    """Create educational diagram explaining what a spectrogram is."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1. Waveform - hard to see frequencies
    sr = 16000
    t = np.linspace(0, 0.5, sr // 2)
    y = np.sin(2 * np.pi * 300 * t) + 0.5 * np.sin(2 * np.pi * 600 * t)

    axes[0].plot(t[:400], y[:400], color='steelblue', linewidth=1)
    axes[0].set_xlabel('Time (s)', fontsize=11)
    axes[0].set_ylabel('Amplitude', fontsize=11)
    axes[0].set_title('Waveform\n(Hard to see frequencies)', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # 2. Frequency spectrum (FFT)
    fft = np.abs(np.fft.fft(y))
    freqs = np.fft.fftfreq(len(y), 1/sr)
    mask = freqs > 0
    axes[1].plot(freqs[mask][:1000], fft[mask][:1000], color='coral', linewidth=1)
    axes[1].axvline(300, color='green', linestyle='--', alpha=0.7, label='300 Hz')
    axes[1].axvline(600, color='purple', linestyle='--', alpha=0.7, label='600 Hz')
    axes[1].set_xlabel('Frequency (Hz)', fontsize=11)
    axes[1].set_ylabel('Magnitude', fontsize=11)
    axes[1].set_title('Frequency Spectrum (FFT)\n(Shows frequencies, loses time)', fontsize=12, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(0, 1500)

    # 3. Spectrogram - changing frequency signal
    t2 = np.linspace(0, 1, sr)
    freq = 200 + 600 * t2  # Chirp: 200 -> 800 Hz
    y2 = np.sin(2 * np.pi * np.cumsum(freq) / sr)
    f, t_spec, Sxx = signal.spectrogram(y2, sr, nperseg=256, noverlap=200)
    axes[2].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
    axes[2].set_xlabel('Time (s)', fontsize=11)
    axes[2].set_ylabel('Frequency (Hz)', fontsize=11)
    axes[2].set_title('Spectrogram\n(Shows frequency over time!)', fontsize=12, fontweight='bold')
    axes[2].set_ylim(0, 1500)

    plt.suptitle('What is a Spectrogram? Frequency × Time = Spectrogram', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'spectrogram_explainer.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: spectrogram_explainer.png")

def create_audio_comparison():
    """Create comparison of original and augmented audio."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    samples = [
        ("original.wav", "Original"),
        ("pitch_up.wav", "Pitch UP (+3 semitones)"),
        ("time_stretch.wav", "Time Stretch (slower)"),
    ]

    for col, (filename, title) in enumerate(samples):
        audio_path = AUDIO_DIR / filename
        if not audio_path.exists():
            print(f"Skipping {filename} - not found")
            continue

        sr, y = wavfile.read(audio_path)
        if y.dtype == np.int16:
            y = y.astype(float) / 32768.0
        if len(y.shape) > 1:
            y = y.mean(axis=1)

        times = np.arange(len(y)) / sr

        # Waveform
        axes[0, col].plot(times, y, color='steelblue', linewidth=0.5)
        axes[0, col].set_title(f'{title}', fontsize=11, fontweight='bold')
        axes[0, col].set_xlabel('Time (s)', fontsize=10)
        if col == 0:
            axes[0, col].set_ylabel('Amplitude', fontsize=10)
        axes[0, col].set_xlim(0, max(times))
        axes[0, col].grid(True, alpha=0.3)

        # Spectrogram
        f, t_spec, Sxx = signal.spectrogram(y, sr, nperseg=256, noverlap=200)
        axes[1, col].pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='magma')
        axes[1, col].set_xlabel('Time (s)', fontsize=10)
        if col == 0:
            axes[1, col].set_ylabel('Frequency (Hz)', fontsize=10)
        axes[1, col].set_ylim(0, min(8000, sr/2))

    # Add row labels
    fig.text(0.02, 0.72, 'Waveform', fontsize=12, fontweight='bold', rotation=90, va='center')
    fig.text(0.02, 0.28, 'Spectrogram', fontsize=12, fontweight='bold', rotation=90, va='center')

    plt.suptitle('Audio Augmentation: See the Difference!', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(left=0.06)
    plt.savefig(OUTPUT_DIR / 'audio_augmentation_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: audio_augmentation_comparison.png")


if __name__ == "__main__":
    print("Generating audio visualizations...")
    create_spectrogram_explainer()
    create_audio_comparison()
    print("\nDone!")
