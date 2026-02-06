import subprocess
import os

def separate_stems(input_wav, output_dir):
    """
    Aegis v3.0: Intelligence Stem Separation.
    Uses Demucs AI to isolate 'Other' (Guitar/Synth) track from a mix.
    """
    print(f"[Aegis] 🎸 Starting Stem Separation: {input_wav}")
    
    demucs_path = "/Users/mac/Library/Python/3.9/bin/demucs"
    try:
        subprocess.run([demucs_path, "--version"], check=True, capture_output=True)
    except:
        print(f"[Aegis] Error: Demucs not found at {demucs_path}. Please install it.")
        return input_wav 
        
    cmd = [demucs_path, "-n", "htdemucs", "-o", output_dir, input_wav]
    subprocess.run(cmd, check=True)
    
    filename = os.path.basename(input_wav).split('.')[0]
    other_path = os.path.join(output_dir, "htdemucs", filename, "other.wav")
    
    if os.path.exists(other_path):
        print(f"[Aegis] Stem Separation Complete: {other_path}")
        return other_path
    else:
        print("[Aegis] Warning: Could not find separated guitar track. Using original.")
        return input_wav
