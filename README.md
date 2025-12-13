# Game-Theoretic Analysis of U.S.-China Economic Relations (2001-2025)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-PhD%20Research%20Edition-purple)

## 📄 Overview

This **PhD-Level Interactive Research Application** employs rigorous game-theoretic frameworks to model the structural transformation of U.S.-China economic relations over the past quarter-century. 

Unlike static academic papers, this application allows researchers, economists, and policymakers to **dynamically explore** the strategic interactions that shifted the bilateral relationship from a cooperative "Harmony Game" (2001-2008) to a conflictual "Prisoner's Dilemma" (2018-2025). It integrates real-world macroeconomic data with sophisticated mathematical models to quantify phenomena such as **"Vendor Financing"** and **Tariff War dynamics**.

---

## 🧭 Navigation & App Structure

The application is structured into **seven core modules**, accessible via the **Sidebar Navigation Menu** on the left.

| Module | Description |
| :--- | :--- |
| **1. Executive Summary** | The landing page. Provides a high-level dashboard with key thesis statements, "Metric Cards" showing live economic data snippets, and a summary of the strategic transition. |
| **2. Strategic Analysis** | The core visualization engine. View the 2x2 Payoff Matrix, explore the 3D Phase Diagrams of the Thucydides Trap, and see the Nash Equilibrium/Pareto Frontier analysis. |
| **3. Mathematical Proofs** | A library of formal derivations. Select a theorem (e.g., "Folk Theorem") to view the step-by-step mathematical proof, intuitive explanation, and empirical validation. |
| **4. Advanced Simulations** | An agent-based modeling hub. Run "Tournaments" or "Evolutionary Simulations" to test how strategies like *Tit-for-Tat* survive against *Always Defect*. |
| **5. Custom Data Workbench** | For data analysts. Select specific datasets (e.g., Trade Deficit vs. 10Y Yields), choose time ranges, and run custom correlation analyses. |
| **6. Methodology** | Academic reference section. Details all data sources, modeling assumptions (rationality, discounting), and calibration techniques. |
| **7. Research Documents** | Digital library. View the embedded PDF User Guide and other attached research papers directly within the app. |

---

## 🔬 Detailed Functionality

### 1. Strategic Analysis Engine
*   **Time-Travel Simulation:** Use the "Select Year" slider to observe how the Payoff Matrix values change from 2001 (High Cooperation Payoff) to 2024 (High Defection Payoff).
*   **Equilibrium Solver:** The app automatically solves for the Nash Equilibrium (NE) for the selected year.
    *   *Visual indicator:* The NE cell is highlighted in the matrix.
    *   *Efficiency check:* The app calculates if the NE is Pareto Efficient.

### 2. Mathematical Proof Explorer
*   **Interactive Parameters:** Many proofs include sliders to adjust variables like the **Discount Factor ($ \delta $)**.
*   **Real-Time Rendering:** See how changing $ \delta $ from 0.9 to 0.4 shifts the "Cooperation Threshold," instantly updating the mathematical inequality on screen.
*   **Empirical Bridging:** Each proof is paired with a "Real-World Evidence" box linking the math to specific historical events (e.g., 2018 Section 301 Tariffs).

### 3. Advanced Simulations Hub
This section runs computationally intensive agent-based models:
*   **Tournament Arena:**
    *   *Setup:* Select a roster of strategies (Random, Grudger, Tit-for-Tat, etc.).
    *   *Run:* Execute a round-robin tournament (Axelrod style).
    *   *Output:* View a ranked leaderboard of total payoffs.
*   **Evolutionary Lab:**
    *   *Dynamics:* Uses **Replicator Dynamics** equations.
    *   *Visualization:* A "Streamgraph" showing how the population share of each strategy evolves over 50+ generations.
    *   *Variables:* Adjust "Noise" (probability of accidental defection) to see if Tit-for-Tat remains robust.

---

## 🛠️ Installation & Local Development

### Prerequisites
*   Python 3.8 or higher.
*   Git.

### Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/us-china-game-theory.git
cd us-china-game-theory
```

### Step 2: Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
Install all required packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
Launch the Streamlit interface:
```bash
streamlit run app.py
```
The application will open automatically in your default web browser at `http://localhost:8501`.

---

## ☁️ Deployment Guide

### Option 1: Streamlit Community Cloud (Recommended)
The easiest way to deploy this app is via Streamlit's free hosting service.

1.  Push your code to a public **GitHub repository**.
2.  Ensure `requirements.txt` is in the root directory.
3.  Log in to [share.streamlit.io](https://share.streamlit.io/).
4.  Click **"New App"**.
5.  Select your repository, branch, and the main file path (`app.py`).
6.  Click **"Deploy"**.

### Option 2: Docker
Building a container for enterprise deployment.

1.  Create a `Dockerfile`:
    ```dockerfile
    FROM python:3.9-slim
    WORKDIR /app
    COPY . .
    RUN pip install -r requirements.txt
    EXPOSE 8501
    CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
    ```
2.  Build and Run:
    ```bash
    docker build -t game-theory-app .
    docker run -p 8501:8501 game-theory-app
    ```

---

## 📚 Theoretical Framework

The analysis is grounded in the following academic literature:

1.  **Nash Equilibrium (Nash, 1950):** Used to identify stable outcome profiles in the static Tariff Game.
2.  **Pareto Efficiency (Pareto, 1906):** Demonstrates the inefficiency of the current "Trade War" equilibrium.
3.  **Repeated Games (Friedman, 1971):** Models the sustainment of free trade through the "Vendor Finance" mechanism (2001-2008).
4.  **Evolution of Cooperation (Axelrod, 1984):** Explains how "Tit-for-Tat" strategies emerged during the 2018 Tariff escalation.

---

## 📊 Data Sources

All empirical data is sourced from official government and international institutions:

| Variable | Source | Frequency |
|----------|--------|-----------|
| **Trade Balance** | U.S. Census Bureau | Annual |
| **FX Reserves** | SAFE (China) | Annual |
| **Treasury Yields** | FRED (St. Louis Fed) | Daily/Annual |
| **Tariff Rates** | Peterson Institute (PIIE) | Event-based |
| **GDP Growth** | World Bank | Annual |

---

## 📜 Citation

If you use this tool for research, please cite:

> **ECON 606 Research Team.** (2025). *Game-Theoretic Analysis of U.S.-China Economic Relations: A Computational Approach.* Version 3.0.0.

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
