# Security Policy

## Scope

This project is an offline inference library: it loads ONNX models and reads image and
PDF files you give it. The realistic risk surface is therefore **malicious input files**
(crafted images or PDFs reaching OpenCV, Pillow, PyMuPDF or onnxruntime) and **model
files from an untrusted source**.

Only download model weights from the official
[Hugging Face repository](https://huggingface.co/rivoklabs/paddleocr-hebrew) and verify
them against the `md5sums.txt` shipped in each model folder.

## Reporting a vulnerability

Please report security issues privately to **ronen@rivoklabs.com** rather than opening a
public issue. Include enough detail to reproduce, and give us a reasonable window to
respond before disclosing publicly.

If the vulnerability is in an upstream dependency (PaddleOCR, onnxruntime, OpenCV,
Pillow, PyMuPDF), report it to that project as well — we can only pin or patch around it.
