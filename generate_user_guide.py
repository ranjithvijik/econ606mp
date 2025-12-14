from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import datetime

def create_comprehensive_guide():
    doc = SimpleDocTemplate("User Guide.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Custom Styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=26,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#1a365d')
    )
    
    subtitle_style = ParagraphStyle(
        'Subtitle',
        parent=styles['Normal'],
        fontSize=14,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#4a5568')
    )

    h2_style = ParagraphStyle(
        'H2Custom',
        parent=styles['Heading2'],
        fontSize=18,
        spaceBefore=20,
        spaceAfter=12,
        textColor=colors.HexColor('#2c5282'),
        borderPadding=(0, 0, 5, 0),
        borderWidth=1,
        borderColor=colors.HexColor('#e2e8f0'),
        borderRadius=None
    )

    h3_style = ParagraphStyle(
        'H3Custom',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=15,
        spaceAfter=8,
        textColor=colors.HexColor('#2b6cb0')
    )

    body_style = ParagraphStyle(
        'BodyCustom',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        alignment=TA_JUSTIFY,
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'Code',
        parent=styles['Code'],
        fontSize=9,
        leading=11,
        backColor=colors.HexColor('#f7fafc'),
        borderColor=colors.HexColor('#cbd5e0'),
        borderWidth=0.5,
        borderPadding=5
    )

    # --- TITLE PAGE ---
    story.append(Spacer(1, 2*inch))
    story.append(Paragraph("Game-Theoretic Analysis of<br/>U.S.-China Economic Relations", title_style))
    story.append(Paragraph("Comprehensive User Guide & Technical Documentation", subtitle_style))
    story.append(Spacer(1, 1*inch))
    story.append(Paragraph(f"Version: 4.0.0<br/>Date: {datetime.date.today().strftime('%B %d, %Y')}", 
                           ParagraphStyle('Center', parent=body_style, alignment=TA_CENTER)))
    story.append(PageBreak())

    # --- TABLE OF CONTENTS ---
    story.append(Paragraph("Table of Contents", h2_style))
    toc_data = [
        ["1. Introduction & Overview", "3"],
        ["2. Navigation & User Interface", "4"],
        ["3. Core Game Theory Engine", "5"],
        ["4. Advanced Simulations", "6"],
        ["   4.1 Tournament Arena", "6"],
        ["   4.2 Evolutionary Dynamics", "7"],
        ["   4.3 Spatial Evolutionary Game", "8"],
        ["   4.4 Gene Lab (Custom Strategy)", "9"],
        ["   4.5 Learning Dynamics", "10"],
        ["   4.6 Stochastic Games", "11"],
        ["5. Statistical Analysis", "12"],
        ["6. Visualization Engine", "13"],
        ["7. Mathematical Proofs Library", "14"],
        ["8. Automated Research Reporting", "15"],
        ["9. Data Sources & Methodology", "16"],
        ["10. Installation & Testing", "17"],
        ["11. Troubleshooting & Support", "18"]
    ]
    t_toc = Table(toc_data, colWidths=[5*inch, 1*inch])
    t_toc.setStyle(TableStyle([
        ('TEXTCOLOR', (0,0), (-1,-1), colors.black),
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 11),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, colors.lightgrey),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # --- 1. INTRODUCTION ---
    story.append(Paragraph("1. Introduction & Overview", h2_style))
    story.append(Paragraph("""
    This application is an interactive research tool designed to analyze the structural transformation 
    of U.S.-China economic relations from 2001 to 2025 using rigorous game-theoretic frameworks. Unlike static 
    reports, this application allows users to manipulate variables, run simulations, and visually explore the 
    mathematical underpinnings of trade conflict.""", body_style))
    
    story.append(Paragraph("""
    <b>Key Capabilities:</b><br/>
    • <b>Nash Equilibrium Analysis:</b> Automatic identification of pure and mixed strategy equilibria.<br/>
    • <b>Pareto Efficiency:</b> Identification of efficient and dominated outcomes.<br/>
    • <b>Folk Theorem Calculations:</b> Critical discount factor δ* computation.<br/>
    • <b>Multi-Agent Simulations:</b> Tournaments, evolutionary dynamics, spatial games, and learning algorithms.<br/>
    • <b>Custom Strategy Design:</b> Build your own agent using Memory-1 DNA parameters.<br/>
    • <b>Statistical Validation:</b> Correlation tests, regression analysis, Monte Carlo simulations.<br/>
    • <b>Automated Reporting:</b> One-click professional HTML report generation.
    """, body_style))
    story.append(PageBreak())

    # --- 2. NAVIGATION ---
    story.append(Paragraph("2. Navigation & User Interface", h2_style))
    story.append(Paragraph("The application uses a Sidebar Navigation menu organized into research categories:", body_style))

    nav_data = [
        ["Category", "Modules", "Description"],
        ["📊 Overview & Documents", "Executive Summary, Methodology, Research Documents", "High-level dashboard, citations, PDF library"],
        ["♟️ Theoretical Frameworks", "Nash Equilibrium, Pareto Efficiency, Repeated Games", "Core game theory analysis"],
        ["🧪 Simulation Laboratory", "Strategy Simulator, Tournament Arena, Evolutionary Lab, Learning Dynamics", "Agent-based modeling tools"],
        ["📈 Empirical Analysis", "Empirical Validation, Advanced Analytics", "Statistical validation with real data"],
        ["📐 Mathematical Tools", "Mathematical Proofs, Parameter Explorer", "26+ proofs, interactive exploration"]
    ]
    t_nav = Table(nav_data, colWidths=[1.5*inch, 2.2*inch, 2.8*inch])
    t_nav.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f7fafc')])
    ]))
    story.append(t_nav)
    story.append(Spacer(1, 20))
    story.append(Paragraph("<b>Theme Toggle:</b> Click '🌓 Toggle Theme' in the sidebar to switch between Light and Dark modes. All charts and UI elements adapt automatically.", body_style))
    story.append(PageBreak())

    # --- 3. CORE GAME THEORY ENGINE ---
    story.append(Paragraph("3. Core Game Theory Engine", h2_style))
    story.append(Paragraph("""
    The <b>GameTheoryEngine</b> class is the mathematical heart of the application. It implements:
    """, body_style))
    
    engine_features = [
        ["Function", "Description"],
        ["find_nash_equilibria()", "Identifies all pure strategy Nash Equilibria using best-response analysis"],
        ["find_mixed_strategy_equilibrium()", "Calculates interior mixed strategy probabilities when applicable"],
        ["find_dominant_strategies()", "Detects strictly dominant strategies for each player"],
        ["classify_game_type()", "Classifies games as PD, Harmony, Stag Hunt, Chicken, or Deadlock"],
        ["pareto_efficiency_analysis()", "Identifies Pareto-efficient and dominated outcomes"],
        ["calculate_critical_discount_factor()", "Computes δ* = (T-R)/(T-P) from Folk Theorem"],
        ["calculate_cooperation_margin(δ)", "Evaluates cooperation sustainability at any discount factor"],
        ["simulate_strategy()", "Runs single-agent simulations with trembling-hand noise"]
    ]
    t_engine = Table(engine_features, colWidths=[2.5*inch, 4*inch])
    t_engine.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (0,-1), 'Courier'),
        ('FONTSIZE', (0,1), (0,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    story.append(t_engine)
    story.append(PageBreak())

    # --- 4. ADVANCED SIMULATIONS ---
    story.append(Paragraph("4. Advanced Simulations", h2_style))
    
    # 4.1 Tournament Arena
    story.append(Paragraph("4.1 Tournament Arena", h3_style))
    story.append(Paragraph("""
    An Axelrod-style round-robin tournament where every strategy competes against every other strategy.
    <br/><br/>
    <b>Available Strategies:</b> Tit-for-Tat, Grim Trigger, Always Cooperate, Always Defect, Pavlov (Win-Stay Lose-Shift), 
    Random, Generous TFT, and <b>Custom Agent</b>.
    <br/><br/>
    <b>Features:</b><br/>
    • Configure rounds per match (10-500)<br/>
    • Add trembling-hand noise (0-20% error probability)<br/>
    • View payoff heatmap, rankings bar chart, head-to-head analysis<br/>
    • Export results to CSV/JSON
    """, body_style))
    
    # 4.2 Evolutionary Dynamics
    story.append(Paragraph("4.2 Evolutionary Dynamics Lab", h3_style))
    story.append(Paragraph("""
    Simulates strategy evolution using <b>Replicator Dynamics</b> equations.
    <br/><br/>
    <b>Mechanism:</b> Strategies reproduce proportionally to their relative fitness (average payoff). 
    A configurable mutation rate introduces random strategy changes.
    <br/><br/>
    <b>Output:</b> Animated streamgraph showing population share evolution over 50+ generations.
    <br/><br/>
    <b>Key Insight:</b> Cooperation survives only when the discount factor δ exceeds the critical threshold δ*.
    """, body_style))
    story.append(PageBreak())
    
    # 4.3 Spatial Evolutionary Game (NEW)
    story.append(Paragraph("4.3 Spatial Evolutionary Game (NEW)", h3_style))
    story.append(Paragraph("""
    <b>NEW FEATURE:</b> Visualizes evolutionary dynamics on a 2D spatial grid.
    <br/><br/>
    <b>How It Works:</b><br/>
    • Agents are placed on an N×N grid (configurable size 10-50).<br/>
    • Each agent plays iterated games with their 8 neighbors (Moore neighborhood).<br/>
    • After each generation, cells adopt the strategy of their highest-performing neighbor.<br/>
    • Successful strategies "spread" across the grid like a contagion.
    <br/><br/>
    <b>Visualization:</b> Animated heatmap with each strategy represented by a distinct color.
    Use the frame slider to step through generations.
    """, body_style))
    
    # 4.4 Gene Lab (NEW)
    story.append(Paragraph("4.4 Gene Lab - Custom Strategy Builder (NEW)", h3_style))
    story.append(Paragraph("""
    <b>NEW FEATURE:</b> Design your own agent using Memory-1 DNA parameters.
    <br/><br/>
    <b>DNA Parameters:</b><br/>
    • <b>P(C|CC):</b> Probability of cooperating after mutual cooperation (Reciprocity)<br/>
    • <b>P(C|CD):</b> Probability of cooperating after being exploited (Forgiveness)<br/>
    • <b>P(C|DC):</b> Probability of cooperating after exploiting opponent (Contrition)<br/>
    • <b>P(C|DD):</b> Probability of cooperating after mutual defection (Optimism)
    <br/><br/>
    <b>Preset Profiles:</b><br/>
    • <i>Tit-for-Tat:</i> (1.0, 0.0, 1.0, 0.0)<br/>
    • <i>Grim Trigger:</i> (1.0, 0.0, 0.0, 0.0)<br/>
    • <i>Generous TFT:</i> (1.0, 0.33, 1.0, 0.0)
    <br/><br/>
    <b>Tools:</b><br/>
    • <b>Radar Chart:</b> Visualizes your strategy's "personality" profile.<br/>
    • <b>Benchmark Button:</b> Test your custom agent against all standard strategies.
    """, body_style))
    story.append(PageBreak())
    
    # 4.5 Learning Dynamics
    story.append(Paragraph("4.5 Learning Dynamics", h3_style))
    story.append(Paragraph("""
    Three learning algorithms for adaptive agents:
    <br/><br/>
    <b>Fictitious Play:</b> Each player best-responds to the empirical distribution of opponent's past actions. 
    Cooperation probabilities converge to Nash mixed strategy over time.
    <br/><br/>
    <b>Reinforcement Learning (Q-Learning):</b> Maintain Q-values for Cooperate/Defect actions. 
    Update based on received payoffs using configurable learning rate.
    <br/><br/>
    <b>Regret Matching:</b> Track cumulative regret for not taking each action. 
    Select actions with probability proportional to positive regret.
    <br/><br/>
    <b>Visualization:</b> Multi-panel charts showing cooperation probability evolution, 
    payoff trajectories, and rolling averages.
    """, body_style))
    
    # 4.6 Stochastic Games
    story.append(Paragraph("4.6 Stochastic Games", h3_style))
    story.append(Paragraph("""
    Model games with state-dependent payoffs:
    <br/><br/>
    <b>States:</b><br/>
    • <b>State 0 (Cooperative):</b> Harmony-like payoffs (high cooperation incentive)<br/>
    • <b>State 1 (Neutral):</b> Mixed payoffs<br/>
    • <b>State 2 (Hostile):</b> Prisoner's Dilemma payoffs (defection dominant)
    <br/><br/>
    The game transitions between states according to a Markov chain with configurable transition probabilities.
    <br/><br/>
    <b>Output:</b> State occupancy chart, payoff evolution by state.
    """, body_style))
    story.append(PageBreak())

    # --- 5. STATISTICAL ANALYSIS ---
    story.append(Paragraph("5. Statistical Analysis Engine", h2_style))
    story.append(Paragraph("""
    The <b>StatisticalEngine</b> class provides rigorous empirical validation:
    """, body_style))
    
    stats_features = [
        ["Method", "Description"],
        ["pearson_correlation_test()", "Correlation coefficient with p-value and 95% CI"],
        ["t_test_correlation()", "Formal hypothesis test for ρ = 0"],
        ["regression_analysis()", "Simple linear regression with R², slope, intercept"],
        ["bootstrap_confidence_interval()", "Non-parametric CI using 10,000 bootstrap samples"],
        ["monte_carlo_simulation()", "Robustness testing across δ ∈ [0.3, 0.9]"]
    ]
    t_stats = Table(stats_features, colWidths=[2.5*inch, 4*inch])
    t_stats.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTNAME', (0,1), (0,-1), 'Courier'),
        ('FONTSIZE', (0,1), (0,-1), 9),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t_stats)
    story.append(PageBreak())

    # --- 6. VISUALIZATION ENGINE ---
    story.append(Paragraph("6. Visualization Engine", h2_style))
    story.append(Paragraph("""
    Professional Plotly-based visualizations with full dark mode support:
    """, body_style))
    
    viz_list = [
        "• <b>Payoff Matrix Heatmaps:</b> Interactive 2×2 game displays with annotations",
        "• <b>Cooperation Margin Charts:</b> δ* threshold with shaded sustainability regions",
        "• <b>3D Payoff Surfaces:</b> Phase diagrams showing game type evolution",
        "• <b>Tournament Heatmaps:</b> Strategy-vs-strategy payoff matrices",
        "• <b>Evolutionary Streamgraphs:</b> Population dynamics over generations",
        "• <b>Spatial Grid Animations:</b> Strategy contagion visualization",
        "• <b>Learning Dynamics Charts:</b> Multi-panel with rolling averages",
        "• <b>Correlation Heatmaps:</b> Variable relationship matrices",
        "• <b>Tariff Escalation Timelines:</b> U.S. and China tariff evolution",
        "• <b>Yield Suppression Charts:</b> Actual vs. counterfactual yields"
    ]
    for item in viz_list:
        story.append(Paragraph(item, body_style))
    story.append(PageBreak())

    # --- 7. MATHEMATICAL PROOFS ---
    story.append(Paragraph("7. Mathematical Proofs Library", h2_style))
    story.append(Paragraph("""
    26+ formal derivations organized into 9 categories:
    """, body_style))
    
    proofs_data = [
        ["Category", "Proofs"],
        ["1. Nash Equilibrium", "Existence, Uniqueness, Dominance Solvability"],
        ["2. Pareto Efficiency", "Efficiency Properties, Frontier Analysis"],
        ["3. Repeated Games", "Folk Theorem, Grim Trigger Sustainability"],
        ["4. Tit-for-Tat Dynamics", "Subgame Perfection, Retaliation Patterns"],
        ["5. Vendor Financing", "Yield Suppression, 'Defection Dividend'"],
        ["6. Discount Factor", "Critical Thresholds, Sensitivity Analysis"],
        ["7. Game Classification", "Type Identification Criteria"],
        ["8. Cooperation Margin", "Erosion Rates, Stability Conditions"],
        ["9. Statistical Correlations", "Pearson Coefficients, Tariff Tests"]
    ]
    t_proofs = Table(proofs_data, colWidths=[2*inch, 4.5*inch])
    t_proofs.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t_proofs)
    story.append(Spacer(1, 20))
    story.append(Paragraph("""
    <b>Interactive Features:</b><br/>
    • Sliders to adjust parameters (e.g., discount factor δ)<br/>
    • Real-time re-rendering of mathematical inequalities<br/>
    • "Real-World Evidence" boxes linking math to historical events<br/>
    • Related proofs navigation
    """, body_style))
    story.append(PageBreak())

    # --- 8. AUTOMATED REPORTING ---
    story.append(Paragraph("8. Automated Research Reporting (NEW)", h2_style))
    story.append(Paragraph("""
    <b>NEW FEATURE:</b> Generate professional HTML research reports with one click.
    <br/><br/>
    <b>How to Use:</b><br/>
    1. Run any simulation or analysis in the application.<br/>
    2. Look for the "📄 Download Report (HTML)" button in the sidebar.<br/>
    3. Click to generate and download.
    <br/><br/>
    <b>Report Contents:</b><br/>
    • Current Payoff Matrix & Game Parameters (T, R, P, S)<br/>
    • Nash Equilibria & Stability Analysis<br/>
    • Critical Discount Factor δ* Calculation<br/>
    • Summary Tables of Simulation Results (Tournament, Evolutionary, Spatial)<br/>
    • KPI Dashboard Values
    <br/><br/>
    <b>Print to PDF:</b> The HTML report includes print-friendly CSS. 
    Open in browser and use File → Print → Save as PDF for academic submission.
    """, body_style))
    story.append(PageBreak())

    # --- 9. DATA SOURCES ---
    story.append(Paragraph("9. Data Sources & Methodology", h2_style))
    story.append(Paragraph("All empirical data is sourced from official government and international institutions:", body_style))
    
    data_sources = [
        ["Variable", "Source", "Frequency"],
        ["U.S. Trade Deficit", "U.S. Census Bureau", "Annual (2001-2024)"],
        ["China FX Reserves", "SAFE (State Admin of Foreign Exchange)", "Annual"],
        ["Treasury Yields (10Y)", "FRED (Federal Reserve)", "Daily/Annual Avg"],
        ["Treasury Holdings", "U.S. Treasury TIC System", "Annual"],
        ["GDP Growth", "World Bank", "Annual"],
        ["Tariff Rates", "Peterson Institute (PIIE)", "Event-based (2018-2025)"],
        ["Yield Suppression", "Warnock & Warnock (2009)", "Derived (-2.4bp/$100B)"]
    ]
    t_data = Table(data_sources, colWidths=[2*inch, 2.8*inch, 1.7*inch])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t_data)
    story.append(PageBreak())

    # --- 10. INSTALLATION ---
    story.append(Paragraph("10. Installation & Testing", h2_style))
    story.append(Paragraph("""
    <b>Prerequisites:</b> Python 3.8+, Git
    <br/><br/>
    <b>Installation Steps:</b>
    """, body_style))
    story.append(Paragraph("""
    <font face="Courier">
    git clone https://github.com/ranjithvijik/econ606mp.git<br/>
    cd econ606mp<br/>
    python -m venv venv<br/>
    source venv/bin/activate<br/>
    pip install -r requirements.txt<br/>
    streamlit run app.py
    </font>
    """, code_style))
    story.append(Spacer(1, 15))
    story.append(Paragraph("""
    <b>Running Tests:</b>
    """, body_style))
    story.append(Paragraph("""
    <font face="Courier">
    python tests/test_app.py
    </font>
    """, code_style))
    story.append(Paragraph("""
    Tests cover: Nash Equilibrium calculations, critical discount factor, tournament execution, 
    evolutionary dynamics, spatial simulation, and custom strategy (Gene Lab) behavior.
    """, body_style))
    story.append(PageBreak())

    # --- 11. TROUBLESHOOTING ---
    story.append(Paragraph("11. Troubleshooting & Support", h2_style))
    story.append(Paragraph("""
    <b>Common Issues:</b>
    <br/><br/>
    • <b>Visualizations Blank:</b> Refresh the page (Ctrl+R). Heavy simulations can temporarily stall Plotly charts.
    <br/><br/>
    • <b>PDFs Not Opening:</b> Ensure your browser allows pop-ups, or use the embedded viewer in Research Documents.
    <br/><br/>
    • <b>Slow Performance:</b> Reduce population size or grid size in simulations. Enable caching (automatic).
    <br/><br/>
    • <b>Import Errors:</b> Ensure you've installed all dependencies: <font face="Courier">pip install -r requirements.txt</font>
    <br/><br/>
    • <b>Dark Mode Issues:</b> If charts appear with wrong colors, toggle theme twice to force refresh.
    <br/><br/>
    <b>Contact:</b> For bugs or feature requests, please open an issue on the GitHub repository.
    """, body_style))

    # Build the document
    doc.build(story)
    print("User Guide.pdf created successfully.")

if __name__ == "__main__":
    create_comprehensive_guide()
