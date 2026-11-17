# Individual Project Proposal

**Student Name:** Manuel Vazhakal
**Student ID:** 74143

## 1. Problem Statement
Web servers create a lot of network logs. This project will build a small machine learning model to quickly spot bad traffic (like hacking attempts or DDoS attacks) to block them before they reach the main database.

## 2. Dataset
I will use the **CSE-CIC-IDS2018** dataset, which contains standard network traffic and security logs.

## 3. Success Metric
The goal is to get an **F1-Score > 0.92**. This makes sure we catch the actual threats without accidentally blocking real users.

## 4. Deployment Constraints
To make sure the model doesn't slow down the web server, it must meet these limits:
* **Latency:** $\le 50$ ms per request.
* **Model Size:** $\le 5$ MB so it fits easily on small servers.

## 5. Planned Optimization
To make the model small and fast enough, I will use **Post-Training Quantization**. I will convert the model weights from standard 32-bit floats down to 8-bit integers. This will make the file size 4 times smaller and speed up the CPU time so it easily runs under 50 ms.