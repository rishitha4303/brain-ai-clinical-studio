# Brain AI Clinical Studio

Streamlit-based clinical decision support app for brain imaging analysis:
- CT hemorrhage classification with Grad-CAM explainability
- MRI tumor segmentation with overlays and metrics
- Severity assessment and report generation

## Tech Stack
- Python
- Streamlit
- TensorFlow/Keras model inference
- OpenCV and NumPy image processing

## Project Structure
- `app_old.py` - current primary UI/app flow
- `app.py`, `new_app.py` - alternate/legacy app variants
- `preprocessing/` - CT and MRI preprocessing pipelines
- `models/` - model loaders and model weight files (`.h5`)
- `xai/` - explainability overlays for CT/MRI
- `utils/` - severity and report utilities
- `requirements.txt` - Python dependencies

## What Was Updated (UI/UX Refinements)
Recent changes focused on clinician dashboard usability while preserving inference logic:

### Hero Scan Preview Collage
- Converted preview tiles to compact square dashboard tiles
- Applied `object-fit: cover` for better image cropping
- Moved tile labels to lightweight in-image overlays
- Tightened spacing and removed bulky outer wrappers for cleaner alignment
- Fixed stray rendered HTML closing tag issue in hero render path

### CT Visualization Section
- Reduced oversized Grad-CAM display for better balance
- Reworked CT visual block to fill right-side empty space with contextual cards
- Added CT Focus Summary card beside Grad-CAM
- Added in-panel heatmap legend card
- Tightened middle layout gap between image and side panel

### Confidence and Severity Visuals
- Confidence switched to a bullet-chart style marker track
- Severity changed to a different variant (stepped Low/Moderate/High strip)
- Kept styles intentionally non-identical for clearer visual hierarchy

## Run Locally
1. Create and activate a virtual environment
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Launch Streamlit:
   - `streamlit run app_old.py`

## Notes
- Model weight files in `models/` are required for full inference.
- If pushing `.h5` files to GitHub fails due to file size, use Git LFS.
