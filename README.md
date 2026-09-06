**Current status:** *Early prototype. I’m building a small ML pipeline to practice model training, evaluation, and eventually integrate workload/energy measurements.*

Abstract

*AI Data centers in North American communities continue to be a rising issue and source of tension between ordinary people and technologists. people feel like they cannot trust AI technology because of the inconsiderate and irresponsible placement of the data centers in locations that put a strain on local communities' water, electricity, and air quality. To potentially solve this issue, this project is aimed to be the basis of research long-term and develop methods for sustainable and ethical computation in AI data centers via nuclear energy. By using nuclear power as a clean energy source, tech corporations can mitigate and reduce its environmental effects and restore trust between AI technology and the people who seek to utilize it. This Hardware-in-the-Loop testbed simulates machine learning & computation processes with a power source derived from an embedded system, a makeshift nuclear power plant with control panels (toggles, dials, displays, etc) and Safety interlocks, alarm states using LEDs/buzzer/buttons, and a Raspberry Pi 3. This system incorporates a small ML model ran on the Pi 3 ([dataset for the ML model](https://www.kaggle.com/datasets/yldrmmahmud/sustainable-ai-model-efficiency-co2-and-energy)) to simulate AI data center processes, and the corresponding embedded systems control allow for the control and adjustment of the 'power supply' from the simulated Nuclear power source and allow for the analysis of computation processes on the Pi 3. via [RocksDB ](https://rocksdb.org/) software.*

README: Nuclear-Aware AI Compute Energy Simulator: A Hardware-in-the-Loop Digital Twin for Sustainable ML Workloads

Code

The code (only BERT and RNN models) are IN PROGRESS of being written and are based on learning resources from HUSAI Bootcamp for highschool students.

References

[HUSAI Bootcamp](https://ai.hcs.harvard.edu/) and Frontiers track [D1C1VC Bootcamp Neural Network](https://colab.research.google.com/drive/1cESttSb2XtU-921VL5NuCl0gJQ7mg51R?authuser=1), [D2C1VC - NLP Classification Student](https://colab.research.google.com/drive/1JJ1fOAgFGNkAuhuEtOWtjeAf3c1VlAZY?authuser=1), [D3C1VC - CNN Solutions](https://colab.research.google.com/drive/1XeY_ctaDpUtYfo6SzoPQB0hGTrHCfIgp?authuser=1)
