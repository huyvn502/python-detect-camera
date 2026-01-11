# Dual-Mode Detection System

Your video processor now supports **TWO MODES**:

## 🎯 Mode 1: Specific Person Match (With Reference)

**Use when:** You want to find a specific person

**Command:**
```bash
python src/main.py --videos videos --reference "ref/person.png"
```

**What it does:**
- Compares each frame to your reference silhouette
- Only reports matches with similar body proportions
- Focus on matching the specific person

---

## 🔍 Mode 2: General Human Detection (No Reference)

**Use when:** You want to detect ANY human in the videos

**Command:**
```bash
python src/main.py --videos videos
```

**What it does:**
- Detects ANY human bodies/shadows in frames
- No comparison to reference image
- Reports any frame containing a human figure

---

## Examples

### Detect specific person:
```bash
source venv/bin/activate
python src/main.py --videos videos --reference "ref/john.png"
```

### Detect any human:
```bash
source venv/bin/activate
python src/main.py --videos videos
```

### Detect humans with custom threshold:
```bash
python src/main.py --videos videos --confidence-threshold 80
```

---

## How It Works

### With Reference Image:
1. Uploads your reference silhouette
2. Compares each video frame against it
3. Focuses on matching body proportions (head, shoulders, torso, legs)
4. Reports only matching individuals

### Without Reference Image:
1. Analyzes each video frame independently  
2. Looks for ANY human features (head, shoulders, torso, etc.)
3. Ignores objects, animals, backgrounds
4. Reports any frame containing a human

---

## Configuration

Both modes use the same settings from `.env`:
- `FRAME_SAMPLE_RATE=0.2` (1 frame every 5 seconds)
- `CONFIDENCE_THRESHOLD=50` (minimum % to report)
- `GEMINI_MODEL=gemini-2.5-flash`

---

## Update Your README

The reference parameter is now **optional**:

```bash
# Old (reference required):
python src/main.py --videos ./videos --reference ./person.jpg

# New (reference optional):
python src/main.py --videos ./videos                    # Detect any human
python src/main.py --videos ./videos --reference ./person.jpg  # Match specific person
```
