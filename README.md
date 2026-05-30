# Accident Detection Agent

A notebook-based pipeline for detecting traffic accidents from video using a 4-stage model:
1) YOLOv8 vehicle detection
2) SimpleTracker multi-object tracking
3) ResNet50 feature extraction
4) LSTM sequence classification

## Contents
- accident-detection-pipeline-cvpr.ipynb: End-to-end pipeline notebook

## Quick Start
1) Place videos in a local folder named "videos" (or set SINGLE_VIDEO_PATH in the notebook).
2) Run the notebook cells in order.
3) Outputs are saved under the "outputs" folder.

## Outputs
- outputs/incident_frames: extracted incident frames
- outputs/agentic_incidents.jsonl: structured incident reports

## Notes
- If you want a single video, set SINGLE_VIDEO_PATH.
- If you use a metadata CSV, set TEST_METADATA_PATH.
