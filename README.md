# Game-Theoretic Analysis of U.S.-China Economic Relations (2001-2025)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Edition-purple)
![Version](https://img.shields.io/badge/Version-4.0.0-orange)

## 📄 Overview

This **Interactive Research Application** employs rigorous game-theoretic frameworks to model the structural transformation of U.S.-China economic relations over the past quarter-century. 

Unlike static academic papers, this application allows researchers, economists, and policymakers to **dynamically explore** the strategic interactions that shifted the bilateral relationship from a cooperative "Harmony Game" (2001-2008) to a conflictual "Prisoner's Dilemma" (2018-2025). It integrates real-world macroeconomic data with sophisticated mathematical models to quantify phenomena such as **"Vendor Financing"** and **Tariff War dynamics**.

---

## ✨ Key Features

| Category | Feature | Description |
| :--- | :--- | :--- |
| **Core Analysis** | Nash Equilibrium Solver | Automatically identifies pure and mixed strategy Nash Equilibria |
| **Core Analysis** | Pareto Efficiency Analysis | Determines Pareto-efficient outcomes and identifies dominated strategies |
| **Core Analysis** | Critical Discount Factor (δ*) | Calculates Folk Theorem thresholds for cooperation sustainability |
| **Simulations** | Tournament Arena | Axelrod-style round-robin competitions between strategies |
| **Simulations** | Evolutionary Lab | Replicator dynamics simulation with mutation |
| **Simulations** | Spatial Evolutionary Game | 2D grid-based strategy evolution with Moore neighborhoods |
| **Simulations** | Gene Lab | Custom strategy builder with Memory-1 DNA parameters |
| **Simulations** | Learning Dynamics | Fictitious Play, Reinforcement Learning, Regret Matching |
| **Simulations** | Stochastic Games | State-dependent payoffs with Markov transitions |
| **Analytics** | Monte Carlo Analysis | Bootstrap confidence intervals and sensitivity testing |
| **Analytics** | Statistical Engine | Pearson correlations, t-tests, regression analysis |
| **Visualization** | 3D Payoff Surfaces | Interactive phase diagrams and Thucydides Trap visualization |
| **Export** | Research Report Generator | One-click HTML reports for academic submission |
| **UX** | Dark Mode | Full theme support for all charts and UI components |

---

## 🧭 Navigation & App Structure

The application is structured into **comprehensive modules**, accessible via the **Sidebar Navigation Menu**.

### Research Navigator Categories

| Category | Modules | Description |
| :--- | :--- | :--- |
| **📊 Overview & Documents** | Executive Summary, Methodology, Research Documents | High-level dashboard, citations, and PDF library |
| **♟️ Theoretical Frameworks** | Nash Equilibrium, Pareto Efficiency, Repeated Games | Core game theory analysis pages |
| **🧪 Simulation Laboratory** | Strategy Simulator, Advanced Simulations Hub, Tournament Arena, Evolutionary Lab, Learning Dynamics | All agent-based modeling tools |
| **📈 Empirical Analysis** | Empirical Validation, Advanced Analytics | Statistical validation with real-world data |
| **📐 Mathematical Tools** | Mathematical Proofs, Parameter Explorer | 26+ formal proofs and interactive parameter exploration |

---

## 🔬 Detailed Functionality

### 1. Core Game Theory Engine (`GameTheoryEngine`)

The mathematical heart of the application implementing:

*   **`find_nash_equilibria()`**: Identifies all pure strategy Nash Equilibria using best-response analysis.
*   **`find_mixed_strategy_equilibrium()`**: Calculates interior mixed strategy probabilities when applicable.
*   **`find_dominant_strategies()`**: Detects strictly dominant strategies for each player.
*   **`classify_game_type()`**: Automatically classifies games as Prisoner's Dilemma, Harmony, Stag Hunt, Chicken, or Deadlock based on T > R > P > S orderings.
*   **`pareto_efficiency_analysis()`**: Identifies Pareto-efficient and dominated outcomes.
*   **`calculate_critical_discount_factor()`**: Computes δ* = (T - R) / (T - P) from the Folk Theorem.
*   **`calculate_cooperation_margin(delta)`**: Evaluates sustainability at any discount factor.
*   **`simulate_strategy(strategy, rounds)`**: Runs single-agent simulations with noise support.

### 2. Advanced Simulation Engine (`AdvancedSimulationEngine`)

Implements sophisticated multi-agent simulations:

#### 2.1 Tournament Arena
*   **Axelrod-style round-robin**: Every strategy plays every other strategy (including itself).
*   **Strategies available**: Tit-for-Tat, Grim Trigger, Always Cooperate, Always Defect, Pavlov, Random, Generous TFT, **Custom Agent**.
*   **Noise support**: Trembling-hand implementation with configurable error probability.
*   **Output**: Payoff matrix heatmap, rankings bar chart, head-to-head analysis.

#### 2.2 Evolutionary Dynamics Lab
*   **Replicator Dynamics**: Strategies reproduce proportionally to relative fitness.
*   **Mutation**: Configurable mutation rate introduces strategy diversity.
*   **Visualization**: Animated streamgraph showing population share evolution over generations.

#### 2.3 Spatial Evolutionary Simulation (NEW!)
*   **2D Grid**: Agents placed on an N×N lattice.
*   **Moore Neighborhood**: Each cell interacts with 8 neighbors.
*   **Evolution Rule**: Cells adopt the strategy of their highest-performing neighbor.
*   **Visualization**: Animated heatmap showing strategy "contagion" across the grid.

#### 2.4 Gene Lab - Custom Strategy Builder (NEW!)
*   **Memory-1 Strategy Design**: Define agent behavior using 4 DNA parameters:
    *   `P(C|CC)`: Probability of cooperating after mutual cooperation
    *   `P(C|CD)`: Probability of cooperating after being exploited
    *   `P(C|DC)`: Probability of cooperating after exploiting opponent
    *   `P(C|DD)`: Probability of cooperating after mutual defection
*   **Radar Chart Visualization**: See your strategy's "personality" profile.
*   **Benchmarking**: Test custom agents against all standard strategies.

#### 2.5 Learning Dynamics
*   **Fictitious Play**: Best-respond to empirical distribution of opponent's actions.
*   **Reinforcement Learning (Q-Learning)**: Action values updated via reward signals.
*   **Regret Matching**: Probability-weighted action selection based on cumulative regret.
*   **Visualization**: Cooperation probability evolution, payoff trajectories, rolling averages.

#### 2.6 Stochastic Games
*   **State-Dependent Payoffs**: Three "mood" states (Cooperative, Neutral, Hostile).
*   **Markov Transitions**: Configurable state transition probabilities.
*   **Output**: State occupancy over time, payoff evolution by state.

### 3. Statistical Analysis Engine (`StatisticalEngine`)

Rigorous empirical validation tools:

*   **`pearson_correlation_test()`**: Correlation with significance testing.
*   **`t_test_correlation()`**: Formal hypothesis test for ρ = 0.
*   **`regression_analysis()`**: Simple linear regression with R², slope, intercept.
*   **`bootstrap_confidence_interval()`**: Non-parametric CI estimation.
*   **`monte_carlo_simulation()`**: Robustness testing across parameter ranges.

### 4. Data Management (`DataManager`)

Pre-loaded empirical datasets with full academic citations:

| Dataset | Source | Years | Variables |
| :--- | :--- | :--- | :--- |
| **Macroeconomic Data** | U.S. Census, SAFE China, FRED | 2001-2024 | Trade Deficit, FX Reserves, 10Y Yields, GDP |
| **Tariff Data** | Peterson Institute (PIIE) | 2018-2025 | U.S. and China tariff rates, event dates |
| **Treasury Holdings** | U.S. Treasury TIC System | 2000-2024 | China's U.S. Treasury holdings |
| **Yield Suppression** | Warnock & Warnock (2009) | 2001-2024 | Counterfactual yield estimates |
| **Federal Debt** | FRED, Treasury Bulletin | 2001-2024 | Total debt, China's share |
| **Cooperation Index** | Author's construction | 2001-2025 | Composite cooperation metric |

### 5. Visualization Engine (`VisualizationEngine` & `AdvancedVisualizationEngine`)

Professional Plotly-based visualizations:

*   **Payoff Matrix Heatmaps**: Interactive 2x2 game displays with annotations.
*   **Cooperation Margin Charts**: δ* threshold visualization with shaded regions.
*   **3D Payoff Surfaces**: Phase diagram showing game type evolution.
*   **Tournament Heatmaps**: Strategy-vs-strategy payoff matrices.
*   **Evolutionary Streamgraphs**: Population dynamics over generations.
*   **Spatial Grid Animations**: Strategy contagion visualization.
*   **Learning Dynamics Charts**: Multi-panel visualizations with rolling averages.
*   **Correlation Heatmaps**: Variable relationship matrices.
*   **Dark Mode Support**: All visualizations adapt to light/dark themes.

### 6. Mathematical Proofs Library

26+ formal derivations organized into 9 categories:

1.  **Nash Equilibrium Analysis**: Existence, uniqueness, dominance solvability
2.  **Pareto Efficiency**: Efficiency properties, frontier analysis
3.  **Repeated Games**: Folk Theorem, Grim Trigger sustainability
4.  **Tit-for-Tat Dynamics**: Subgame perfection, retaliation patterns
5.  **Vendor Financing**: Yield suppression, "Defection Dividend"
6.  **Discount Factor**: Critical thresholds, sensitivity analysis
7.  **Game Classification**: Type identification criteria
8.  **Cooperation Margin**: Erosion rates, stability conditions
9.  **Statistical Correlations**: Pearson coefficients, tariff correlation tests

### 7. Automated Research Reporting (NEW!)

*   **One-Click Export**: Generate professional HTML reports.
*   **Contents**: Game parameters, Nash Equilibria, simulation summaries, statistical tables.
*   **Print-Friendly**: CSS optimized for PDF conversion.
*   **Sidebar Access**: "Download Report (HTML)" button always available.

### 8. User Experience Features

*   **Dark Mode Toggle**: Full theme support with consistent styling.
*   **Session State Persistence**: Simulation results preserved across page navigation.
*   **Export Buttons**: CSV, JSON, and summary downloads for all data tables.
*   **Quick Start Presets**: One-click tournament configurations (Classic Axelrod, Noisy, Evolutionary).
*   **Interactive Parameter Explorer**: Real-time game classification and equilibrium updates.

---

## 🛠️ Installation & Local Development

### Prerequisites
*   Python 3.8 or higher
*   Git

### Step 1: Clone the Repository
```bash
git clone https://github.com/ranjithvijik/econ606mp.git
cd econ606mp
```

### Step 2: Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```
The application will open at `http://localhost:8501`.

---

## 🧪 Testing

Run the automated test suite:

```bash
python tests/test_app.py
```

Tests cover:
*   Nash Equilibrium calculations
*   Critical discount factor computation
*   Tournament execution
*   Evolutionary dynamics
*   Spatial simulation logic
*   Custom strategy (Gene Lab) behavior

---

## ☁️ Deployment

### Streamlit Community Cloud (Recommended)
1.  Push code to GitHub.
2.  Ensure `requirements.txt` is in root.
3.  Log in to [share.streamlit.io](https://share.streamlit.io).
4.  Select repository and deploy.

### Docker
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 📚 Theoretical Framework

| Concept | Reference | Application |
| :--- | :--- | :--- |
| Nash Equilibrium | Nash (1950) | Identifying stable tariff outcomes |
| Pareto Efficiency | Pareto (1906) | Demonstrating Trade War inefficiency |
| Repeated Games | Friedman (1971), Fudenberg & Maskin (1986) | Modeling cooperation via "Vendor Finance" |
| Evolution of Cooperation | Axelrod (1984) | Tournament analysis, TFT emergence |
| Yield Suppression | Warnock & Warnock (2009) | Quantifying Treasury purchase effects |

---

## 📊 Data Sources

| Variable | Source | Frequency |
|----------|--------|-----------|
| **Trade Balance** | U.S. Census Bureau | Annual |
| **FX Reserves** | SAFE (China) | Annual |
| **Treasury Yields** | FRED (St. Louis Fed) | Daily/Annual |
| **Treasury Holdings** | U.S. Treasury TIC System | Annual |
| **Tariff Rates** | Peterson Institute (PIIE) | Event-based |
| **GDP Growth** | World Bank | Annual |

---

## 📜 Citation

If you use this tool for research, please cite:

> **ECON 606 Research Team.** (2025). *Game-Theoretic Analysis of U.S.-China Economic Relations: A Computational Approach.* Version 4.0.0.

---

## 📂 Project Structure

```
econ606mp/
├── app.py                          # Main Streamlit application (12,000+ lines)
├── generate_user_guide.py          # PDF documentation generator
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── User Guide.pdf                  # Comprehensive user documentation
├── tests/
│   └── test_app.py                 # Automated test suite
├── assets/                         # Static assets
├── ECON 606 Mini Project Report.pdf
└── ECON 606 Mini Project Presentation.pdf
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or submit a pull request for:
*   Bug fixes
*   New strategy implementations
*   Additional empirical datasets
*   Documentation improvements

---

*Last Updated: December 2025 | Version 4.0.0 - Enhanced Simulation Edition*
