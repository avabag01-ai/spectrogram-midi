import subprocess
import os
import sys

def test_separation():
    demucs_path = "/Users/mac/Library/Python/3.9/bin/demucs"
    input_wav = "/Users/mac/.gemini/antigravity/scratch/aegis_engine/danbal.mp3"
    output_dir = "/Users/mac/.gemini/antigravity/scratch/aegis_engine/stem_test_output"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Testing Demucs with: {input_wav}")
    cmd = [demucs_path, "-n", "htdemucs", "-o", output_dir, input_wav]
    
    try:
        # We use check=True to raise error if it fails
        # We also capture output to see details
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
    except subprocess.CalledProcessError as e:
        print("FAILED!")
        print("EXIT CODE:", e.returncode)
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return

    filename = os.path.basename(input_wav).split('.')[0]
    expected_path = os.path.join(output_dir, "htdemucs", filename, "other.wav")
    
    if os.path.exists(expected_path):
        print(f"SUCCESS! Separated file found at: {expected_path}")
    else:
        print(f"FAILURE! File not found at {expected_path}")
        # List files in output_dir to see what happened
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                print(os.path.join(root, file))

if __name__ == "__main__":
    test_separation()
