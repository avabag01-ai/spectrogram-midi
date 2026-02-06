def generate_tabs(events):
    """
    Anti-Gravity Fingerboard Optimizer:
    Calculates the most efficient string/fret positions to minimize physical reach.
    """
    string_pitches = [64, 59, 55, 50, 45, 40] # Standard E Tuning
    tab_data = []
    
    # We use a state-tracking heuristic for the 'Center of Gravity' (avg fret position)
    fret_center = 5 
    
    for evt in events:
        pitch = evt['note']
        # Find all possible string/fret combinations for this pitch
        candidates = []
        for s_idx, s_pitch in enumerate(string_pitches):
            fret = pitch - s_pitch
            if 0 <= fret <= 24: # Up to 24 frets
                candidates.append((s_idx + 1, fret))
        
        if not candidates: continue
        
        # Scoring: minimize distance from previous 'center' + prioritize lower strings for lower notes
        best_cand = min(candidates, key=lambda x: abs(x[1] - fret_center) * 1.5 + x[0] * 0.2)
        
        # Update center of gravity (leaking filter)
        fret_center = (fret_center * 0.7) + (best_cand[1] * 0.3)
        
        tab_data.append({
            'time': evt['start'], 
            'string': best_cand[0], 
            'fret': best_cand[1], 
            'note': pitch, 
            'technique': evt.get('technique'),
            'm_start': evt['start'], 
            'm_end': evt['end']
        })
    return tab_data

def export_musicxml(tab_data, output_path):
    """
    Aegis Professional: MusicXML Exporter for Guitar Pro & Sibelius.
    Encodes String/Fret data into a standard notation format.
    """
    import xml.etree.ElementTree as ET
    
    score = ET.Element("score-partwise", version="3.1")
    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = "Aegis Guitar"
    
    part = ET.SubElement(score, "part", id="P1")
    measure = ET.SubElement(part, "measure", number="1")
    
    # Attributes (Standard Tuning)
    attr = ET.SubElement(measure, "attributes")
    ET.SubElement(attr, "divisions").text = "1"
    key = ET.SubElement(attr, "key")
    ET.SubElement(key, "fifths").text = "0"
    time = ET.SubElement(attr, "time")
    ET.SubElement(time, "beats").text = "4"
    ET.SubElement(time, "beat-type").text = "4"
    
    clef = ET.SubElement(attr, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"
    
    staff_details = ET.SubElement(attr, "staff-details")
    ET.SubElement(staff_details, "staff-lines").text = "6"
    
    # Add notes
    for t in tab_data:
        note = ET.SubElement(measure, "note")
        pitch = ET.SubElement(note, "pitch")
        
        # Simple MIDI to Note Name logic for XML
        step_map = {0: 'C', 1: 'C', 2: 'D', 3: 'D', 4: 'E', 5: 'F', 6: 'F', 7: 'G', 8: 'G', 9: 'A', 10: 'A', 11: 'B'}
        alter_map = {1: 1, 3: 1, 6: 1, 8: 1, 10: 1}
        
        pitch_val = t['note']
        octave = (pitch_val // 12) - 1
        step = step_map[pitch_val % 12]
        alter = alter_map.get(pitch_val % 12, 0)
        
        ET.SubElement(pitch, "step").text = step
        if alter: ET.SubElement(pitch, "alter").text = "1"
        ET.SubElement(pitch, "octave").text = str(octave)
        
        ET.SubElement(note, "duration").text = "1"
        ET.SubElement(note, "type").text = "quarter"
        
        # THE KILLER PART: Technical notation (String & Fret)
        tech = ET.SubElement(note, "notations")
        tech_detail = ET.SubElement(tech, "technical")
        ET.SubElement(tech_detail, "string").text = str(t['string'])
        ET.SubElement(tech_detail, "fret").text = str(t['fret'])
        
        # Add Articulation Symbols
        if t.get('technique'):
            if t['technique'] == 'bend':
                bend = ET.SubElement(tech_detail, "bend")
                ET.SubElement(bend, "bend-alter").text = "2" # Whole step jump
            elif t['technique'] == 'slide':
                ET.SubElement(tech, "slur", type="start", number="1")
            elif t['technique'] == 'vibrato':
                ET.SubElement(tech_detail, "hammer-on", type="start") # Fallback for GP
                ornaments = ET.SubElement(tech, "ornaments")
                ET.SubElement(ornaments, "wavy-line", type="start", number="1")
        
    tree = ET.ElementTree(score)
    tree.write(output_path, encoding="UTF-8", xml_declaration=True)
    return output_path

