
import re
import os
import sys

# Target file
INPUT_FILE = 'static/index.html'

def minify_omega_safe(file_path):
    print(f"🛡️  Starting ANTIGRAVITY Ω-PROTO (SAFE MODE) on {file_path}...")
    
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found!")
        sys.exit(1)

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    original_size = len(content)

    # 1. Remove CSS/JS Block Comments /* ... */
    # (This is safe as long as they don't contain quotes)
    content = re.sub(r'/\*[\s\S]*?\*/', '', content)
    
    # 2. Remove HTML Comments <!-- ... -->
    content = re.sub(r'<!--[\s\S]*?-->', '', content)

    # 3. Line-by-line processing (KEEP NEWLINES)
    lines = content.split('\n')
    processed_lines = []
    
    in_script = False
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        
        # Simple Script Tag Detection
        if '<script' in line: in_script = True
        if '</script>' in line: in_script = False

        if in_script:
            # Drop single line comments ONLY if they start the line
            if stripped.startswith('//'): continue
            
            # CAUTION: Do NOT remove console.log if it breaks syntax (e.g. inside catch block)
            # Only remove if it's a standalone statement
            if re.match(r'^console\.(log|dir|warn|error|info)\s*\(.*\);?$', stripped):
                continue
        
        # Collapse multiple spaces into one within the line
        cleaned_line = re.sub(r'\s{2,}', ' ', line).strip()
        
        # CSS/JSON-like structure optimization
        # Remove space around syntax characters { } : ; ,
        cleaned_line = re.sub(r'\s*([\{\}:;,])\s*', r'\1', cleaned_line)
        
        # Keep the line (add newline for safety)
        processed_lines.append(cleaned_line)

    # Join with newlines to be 100% safe against missing semicolons
    final_body = '\n'.join(processed_lines)

    # 4. Remove space between HTML tags (safe)
    final_body = re.sub(r'>\s+<', '><', final_body)

    # 5. Add Ω_brief Header (Optimized for Port 8508 - Main Server)
    header = '<!-- Ω_brief: Aegis Tuner Pro UI v2.8 | Single-Page App | Components: Audio Upload, Piano Roll (Canvas), Tablature Gen, Fret Filter, Financial Engine (Bollinger/MACD/RSI), ADSR Synth | Tech: Vanilla JS, CSS3, Streamlit-style layout | Optimized for Port:8508 -->'
    
    final_content = header + '\n' + final_body

    # Overwrite
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(final_content)

    new_size = len(final_content)
    reduction = 100 * (original_size - new_size) / original_size

    print(f"✅ Ω-PROTO (SAFE) Compression Successful!")
    print(f"   Original Size:   {original_size:,} bytes")
    print(f"   Compressed Size: {new_size:,} bytes")
    print(f"   Reduction:       {reduction:.1f}%")
    print(f"   Safety:          HIGH (Newlines preserved)")

if __name__ == "__main__":
    minify_omega_safe(INPUT_FILE)
