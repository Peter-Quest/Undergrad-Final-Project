# Integrative Human and Object-Aware Online Progress Observation for Human-Centric Augmented Reality Assembly

<p align="center">
  <a href="https://www.sciencedirect.com/science/article/abs/pii/S1474034624007328">
    <img src="https://img.shields.io/badge/DOI-10.1016%2Fj.aei.2024.103081-blue" alt="DOI">
  </a>
  <a href="https://www.sciencedirect.com/science/article/abs/pii/S1474034624007328">
    <img src="https://img.shields.io/badge/Journal-Advanced%20Engineering%20Informatics-green" alt="Journal">
  </a>
  <img src="https://img.shields.io/badge/SCI-Indexed-orange" alt="SCI">
  <img src="https://img.shields.io/badge/Published-2025-brightgreen" alt="Published">
</p>

## 📖 About

This repository contains my **Undergraduate Final Project**, which was later refined into a research paper published in the **SCI journal _[Advanced Engineering Informatics](https://www.sciencedirect.com/journal/advanced-engineering-informatics)_** (Elsevier, IF: 8.0).

**Authors:** Tienong Zhang, **Yuqing Cui**, Wei Fang  
**Affiliation:** Beijing University of Posts and Telecommunications (BUPT)

---

## 🔬 Abstract

In augmented reality (AR)-assisted assembly, real-time progress observation is essential but challenging due to the need for simultaneous awareness of both human actions and object states. Existing approaches typically focus on either human activity recognition or object detection independently, lacking an integrative framework.

**To alleviate these limitations, this paper proposes a real-time two-branch approach that integrates human action-based human factor evaluation and object-based assembly progress observation.** The method enables:

- 🧍 **Human-aware branch**: Evaluates human factors (fatigue, posture, attention) through action recognition to ensure worker well-being during assembly
- 📦 **Object-aware branch**: Tracks assembly component states and progress through object detection and state estimation
- 🔄 **Integrative fusion**: Combines both branches for comprehensive, online progress observation in human-centric AR assembly scenarios

---

## 🏗️ System Overview

```
┌─────────────────────────────────────────────────────────┐
│              AR Assembly Environment                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│   ┌───────────────┐          ┌───────────────────┐      │
│   │  Human-Aware  │          │   Object-Aware    │      │
│   │    Branch     │          │     Branch        │      │
│   │               │          │                   │      │
│   │ • Action      │          │ • Object          │      │
│   │   Recognition │          │   Detection       │      │
│   │ • Human Factor│          │ • State           │      │
│   │   Evaluation  │          │   Estimation      │      │
│   └───────┬───────┘          └────────┬──────────┘      │
│           │                           │                  │
│           └───────────┬───────────────┘                  │
│                       │                                  │
│           ┌───────────▼───────────┐                      │
│           │  Integrative Fusion   │                      │
│           │  & Progress Observer  │                      │
│           └───────────┬───────────┘                      │
│                       │                                  │
│           ┌───────────▼───────────┐                      │
│           │   Real-time AR        │                      │
│           │   Progress Display    │                      │
│           └───────────────────────┘                      │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 🎯 Key Contributions

1. **Integrative two-branch architecture** that simultaneously considers human actions and object states for AR assembly progress observation
2. **Human-centric design** incorporating human factor evaluation (fatigue, attention, ergonomics) alongside task progress tracking
3. **Real-time online processing** enabling immediate feedback in AR-guided assembly scenarios
4. **Comprehensive experiments** demonstrating the effectiveness of the proposed approach in practical assembly tasks

---

## 🔑 Keywords

`Augmented Reality` · `Assembly Progress Observation` · `Human Action Recognition` · `Object Detection` · `Human-Centric Computing` · `Human Factor Evaluation` · `Smart Manufacturing`

---

## 📄 Paper Access

| Format | Link |
|--------|------|
| 📗 Online (ScienceDirect) | [Read Online](https://www.sciencedirect.com/science/article/abs/pii/S1474034624007328) |
| 📕 PDF (this repo) | [Download PDF](./Integrative%20human%20and%20object%20aware%20online%20progress%20observation%20for%20human-centric%20augmented%20reality%20assembly.pdf) |

---

## 📚 Citation

If you find this work useful, please consider citing:

```bibtex
@article{zhang2025integrative,
  title={Integrative human and object aware online progress observation for human-centric augmented reality assembly},
  author={Zhang, Tienong and Cui, Yuqing and Fang, Wei},
  journal={Advanced Engineering Informatics},
  volume={64},
  pages={103081},
  year={2025},
  publisher={Elsevier},
  doi={10.1016/j.aei.2024.103081}
}
```

---

## 🙏 Acknowledgments

This project was completed under the guidance of Prof. **Wei Fang** at Beijing University of Posts and Telecommunications, with collaboration from **Tienong Zhang** and **Mohan**.

---

## 📬 Contact

For questions about this paper, feel free to reach out:
- **Email**: YuqingCui2001@gmail.com
- **GitHub**: [@Peter-Quest](https://github.com/Peter-Quest)
