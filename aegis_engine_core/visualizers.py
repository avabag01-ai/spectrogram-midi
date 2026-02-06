import streamlit as st
import base64
import mido
import io

def render_python_vector_engine(midi_base64, height, theme="beige"):
    """
    Engine 4: Aegis Pure Python Vector Engine.
    100% Local. No CDNs. No JS Libraries.
    Parses MIDI using mido and generates SVG directly.
    """
    try:
        # 1. Parse MIDI Data Locally
        midi_bytes = base64.b64decode(midi_base64)
        mid = mido.MidiFile(file=io.BytesIO(midi_bytes))
        
        # 2. Architectures & Constants
        bg_color = "#F5F5DC" if theme == "beige" else "#121417"
        grid_color = "rgba(0,0,0,0.15)" if theme == "beige" else "rgba(255,255,255,0.1)"
        bar_color = "rgba(0,0,0,0.3)" if theme == "beige" else "rgba(255,255,255,0.25)"
        note_color = "#4a90e2" if theme == "beige" else "#ff00cc"
        text_color = "#8b4513" if theme == "beige" else "#ff00cc"
        
        # 3. Data Extraction
        notes = []
        max_tick = 0
        for track in mid.tracks:
            curr_tick = 0
            active_notes = {}
            for msg in track:
                curr_tick += msg.time
                if msg.type == 'note_on' and msg.velocity > 0:
                    active_notes[msg.note] = curr_tick
                elif (msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0)) and msg.note in active_notes:
                    start_tick = active_notes.pop(msg.note)
                    notes.append({
                        'pitch': msg.note,
                        'start': start_tick,
                        'end': curr_tick,
                        'velocity': getattr(msg, 'velocity', 64)
                    })
            if curr_tick > max_tick: max_tick = curr_tick
            
        if not notes: return "<div style='color:red;'>No MIDI notes detected.</div>"

        # 4. Viewport Calc
        min_pitch = min(n['pitch'] for n in notes) - 2
        max_pitch = max(n['pitch'] for n in notes) + 2
        pitch_range = max(12, max_pitch - min_pitch)
        
        view_width = 1000
        view_height = height - 40
        tick_scale = view_width / max_tick if max_tick > 0 else 1
        pitch_scale = view_height / pitch_range
        
        # 5. Build SVG String
        svg_parts = [
            f'<svg width="100%" height="{view_height}" viewBox="0 0 {view_width} {view_height}" xmlns="http://www.w3.org/2000/svg" style="background:{bg_color}; border-radius:12px; border:1px solid #d2b48c;">'
        ]
        
        # LAYER: Grid
        # Horizontal Pitch Grid
        for p in range(int(min_pitch), int(max_pitch) + 1):
            y = view_height - (p - min_pitch) * pitch_scale
            svg_parts.append(f'<line x1="0" y1="{y}" x2="{view_width}" y2="{y}" stroke="{grid_color}" stroke-width="0.5" />')
            
        # Vertical Time Grid (Approximate beats)
        ticks_per_beat = mid.ticks_per_beat
        for t in range(0, max_tick, ticks_per_beat):
            x = t * tick_scale
            color = bar_color if (t // ticks_per_beat) % 4 == 0 else grid_color
            svg_parts.append(f'<line x1="{x}" y1="0" x2="{x}" y2="{view_height}" stroke="{color}" stroke-width="1" />')

        # LAYER: Notes
        for n in notes:
            x = n['start'] * tick_scale
            w = (n['end'] - n['start']) * tick_scale
            y = view_height - (n['pitch'] - min_pitch + 1) * pitch_scale
            h = pitch_scale - 1
            
            # Note with slight opacity based on velocity
            op = 0.5 + (n['velocity'] / 127) * 0.5
            svg_parts.append(f'<rect x="{x}" y="{y}" width="{max(2, w)}" height="{max(2, h)}" fill="{note_color}" fill-opacity="{op}" rx="2" stroke="white" stroke-width="0.3" />')
            
        svg_parts.append('</svg>')
        
        # Time Ruler Overlay Concept (Pure HTML/CSS)
        ruler_html = f"""
        <div style="background:{bg_color}; border:1px solid #d2b48c; border-radius:12px; padding:10px; overflow:hidden;">
            <div style="font-family:monospace; font-size:10px; color:{text_color}; margin-bottom:5px; display:flex; justify-content:space-between;">
                <span>AEGIS PURE LOCAL PYTHON ENGINE v4.0 (VECTOR)</span>
                <span>TICKS: {max_tick} | NOTES: {len(notes)}</span>
            </div>
            {"".join(svg_parts)}
        </div>
        """
        return ruler_html

    except Exception as e:
        return f"<div style='color:red; background:white; padding:10px;'>Local Engine Error: {str(e)}</div>"

def render_tonejs_engine(midi_base64, height, theme="beige"):
    bg_color = "#F5F5DC" if theme == "beige" else "#121417"
    grid_color = "rgba(0,0,0,0.1)" if theme == "beige" else "rgba(255,255,255,0.08)"
    text_color = "#8b4513" if theme == "beige" else "#ff00cc"
    return f"""
    <div style="background:{bg_color}; padding:15px; border-radius:12px; border:1px solid #d2b48c;">
        <script src="https://cdn.jsdelivr.net/combine/npm/html-midi-player@1.5.0/dist/midi-player.min.js,npm/html-midi-player@1.5.0/dist/midi-visualizer.min.js"></script>
        <midi-player id="p1" src="data:audio/midi;base64,{midi_base64}" sound-font visualizer="#v1"></midi-player>
        <midi-visualizer id="v1" type="piano-roll" src="data:audio/midi;base64,{midi_base64}" style="width:100%; height:{height-180}px; margin-top:10px;"></midi-visualizer>
    </div>
    <style>
        midi-player {{ width: 100%; --background-color: {'#e6dfcc' if theme == 'beige' else '#1a1d23'}; --color: {text_color}; }}
        midi-visualizer {{ background: {bg_color} !important; display: block; width: 100%; border-radius: 8px; }}
        midi-visualizer svg line.grid-line {{ stroke: {grid_color} !important; stroke-width: 1px !important; }}
        midi-visualizer svg rect.note {{ fill: #4a90e2; rx: 2px; stroke: rgba(0,0,0,0.2); stroke-width: 0.5px; }}
        midi-visualizer svg rect.note.active {{ fill: #d0021b !important; filter: drop-shadow(0 0 8px #d0021b); }}
    </style>
    """

def render_canvas_engine(midi_base64, height, theme="beige"):
    bg_color = "#F5F5DC" if theme == "beige" else "#121417"
    grid_color = "rgba(0,0,0,0.15)" if theme == "beige" else "rgba(255,255,255,0.1)"
    bar_color = "rgba(0,0,0,0.3)" if theme == "beige" else "rgba(255,255,255,0.25)"
    note_color = "#4a90e2" if theme == "beige" else "#ff00cc"
    text_color = "#8b4513" if theme == "beige" else "#ff00cc"
    return f"""
    <div id="canvas-wrapper" style="background:{bg_color}; border:1px solid #d2b48c; border-radius:12px; overflow:hidden; position:relative;">
        <div id="time-ruler" style="height:25px; background:rgba(0,0,0,0.05); border-bottom:1px solid {grid_color}; display:flex; align-items:center; padding:0 10px; font-size:9px; color:{text_color}; font-family:monospace;">TIME RULER (MEASURES/BEATS)</div>
        <canvas id="pianoCanvas" style="width:100%; height:{height-30}px; display:block;"></canvas>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/@tonejs/midi"></script>
    <script>
        async function draw() {{
            const canvas = document.getElementById('pianoCanvas');
            const ctx = canvas.getContext('2d');
            const midiData = "{midi_base64}";
            const midi = new Midi(Uint8Array.from(atob(midiData), c => c.charCodeAt(0)));
            const dpr = window.devicePixelRatio || 1;
            canvas.width = canvas.clientWidth * dpr;
            canvas.height = canvas.clientHeight * dpr;
            ctx.scale(dpr, dpr);
            const duration = midi.duration;
            const width = canvas.clientWidth;
            const height = canvas.clientHeight;
            ctx.fillStyle = '{bg_color}';
            ctx.fillRect(0, 0, width, height);
            const keys = 88;
            const keyHeight = height / keys;
            ctx.strokeStyle = '{grid_color}';
            ctx.lineWidth = 0.5;
            for(let i=0; i <= keys; i++) {{
                const y = height - i * keyHeight;
                ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
            }}
            const totalBeats = duration * (120/60);
            for(let i=0; i < totalBeats; i++) {{
                const x = (i / totalBeats) * width;
                ctx.strokeStyle = (i % 4 == 0) ? '{bar_color}' : '{grid_color}';
                ctx.lineWidth = 1;
                ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
            }}
            midi.tracks.forEach(track => {{
                track.notes.forEach(note => {{
                    const x = (note.time / duration) * width;
                    const w = (note.duration / duration) * width;
                    const y = height - (note.midi - 21 + 1) * keyHeight;
                    const h = Math.max(2, keyHeight - 1);
                    ctx.fillStyle = '{note_color}';
                    ctx.globalAlpha = 0.9;
                    ctx.beginPath(); ctx.roundRect(x, y, w, h, 2); ctx.fill();
                }});
            }});
        }}
        setTimeout(draw, 150);
    </script>
    """

def render_webaudiofont_engine(midi_base64, height, theme="beige"):
    bg_color = "#F5F5DC" if theme == "beige" else "#121417"
    text_color = "#8b4513" if theme == "beige" else "#ff00cc"
    return f"""
    <div style="background:#eadfb4; padding:20px; border-radius:12px; height:{height}px; border:1px solid #d2b48c;">
        <h4 style="color:{text_color}; margin:0 0 10px 0; font-family:monospace;">WEBAUDIOFONT ARCHITECTURE</h4>
        <svg id="waf-svg" width="100%" height="{height-80}px" style="background:{bg_color}; border-radius:8px; border:1px solid #d2b48c;"></svg>
    </div>
    """

def render_vector_piano_roll(midi_data_base64, update_key=None, height=500, engine="python", theme="beige", **kwargs):
    """
    AEGIS MULTI-ENGINE COMMAND: Swaps between 4 core piano roll architectures.
    *Optimized for Pure Local Execution*
    """
    if engine == "python":
        content = render_python_vector_engine(midi_data_base64, height, theme)
        return st.write(content, unsafe_allow_html=True)
    elif engine == "tonejs":
        content = render_tonejs_engine(midi_data_base64, height, theme)
    elif engine == "canvas":
        content = render_canvas_engine(midi_data_base64, height, theme)
    else:
        content = render_webaudiofont_engine(midi_data_base64, height, theme)
        
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head><meta charset="UTF-8"></head>
    <body style="margin:0; background:{'#F5F5DC' if theme == 'beige' else '#000'}; overflow:hidden;">
        {content}
    </body>
    </html>
    """
    return st.components.v1.html(full_html, height=height)
