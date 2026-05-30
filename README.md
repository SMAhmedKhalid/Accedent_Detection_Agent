# Accident Detection Agent

This repository contains a notebook-based pipeline that detects traffic accidents from video and produces structured, explainable incident reports. The workflow combines classical computer vision (CV), deep learning, and an optional local language model to summarize incidents for downstream use.

## What This Project Does
1) Finds vehicles in video frames with YOLOv8.
2) Tracks detected vehicles over time with a lightweight tracker.
3) Extracts visual features with ResNet50.
4) Classifies short video snippets with an LSTM to infer accident type and timing.
5) Builds an "agentic" incident report that includes severity, risk, and recommended notifications.
6) Optionally uses a local LLM (Mistral) to produce a concise, structured summary.

## Computer Vision Pipeline
The CV portion of the notebook is responsible for detecting and localizing the incident.
- **Detection**: YOLOv8 identifies vehicles per frame.
- **Tracking**: SimpleTracker maintains vehicle identities to stabilize features across frames.
- **Feature extraction**: ResNet50 encodes cropped vehicle regions into embeddings.
- **Sequence modeling**: An LSTM ingests a short sequence of embeddings to classify accident type.
- **Incident frame capture**: The pipeline grabs the predicted incident timestamp frame and saves it as a preview image.

## Agentic Reasoning Layer
After the CV pipeline produces raw predictions, an agent-style layer turns them into structured, explainable outputs:
1) **Incident extraction**: Normalize the prediction into a canonical event object.
2) **Severity estimation**: Heuristic scoring based on type, confidence, and involved vehicles.
3) **Risk analysis**: Estimate injury risk and road blockage likelihood.
4) **Decision making**: Recommend which responders to notify.
5) **Reasoning**: Provide concise justifications for the decision.
6) **Report generation**: Optionally call Mistral to generate a brief, structured summary.

This design keeps the outputs interpretable and machine-consumable while still providing human-readable summaries.

## Contents
- accident-detection-pipeline-cvpr.ipynb: End-to-end pipeline notebook

## Quick Start
1) Place videos in a local folder named "videos" (or set SINGLE_VIDEO_PATH in the notebook).
2) Run the notebook cells in order.
3) Outputs are saved under the "outputs" folder.

## Inputs
- **SINGLE_VIDEO_PATH**: Path to a single video file.
- **TEST_METADATA_PATH**: Optional CSV containing video paths or identifiers.
- **videos/**: Folder for batch processing if no CSV is provided.

## Outputs
- outputs/incident_frames: extracted incident frames
- outputs/agentic_incidents.jsonl: structured incident reports and summaries

## Notes
- If you want a single video, set SINGLE_VIDEO_PATH.
- If you use a metadata CSV, set TEST_METADATA_PATH.
- The LLM summary step is optional; the pipeline still produces a deterministic report without it.
