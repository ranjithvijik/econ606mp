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
    story.append(Paragraph(f"Version: 3.0.0<br/>Date: {datetime.date.today().strftime('%B %d, %Y')}", 
                           ParagraphStyle('Center', parent=body_style, alignment=TA_CENTER)))
    story.append(PageBreak())

    # --- TABLE OF CONTENTS (Simulated) ---
    story.append(Paragraph("Table of Contents", h2_style))
    toc_data = [
        ["1. Introduction & Overview", "3"],
        ["2. Navigation & User Interface", "4"],
        ["3. Core Functional Modules", "5"],
        ["   3.1 Strategic Analysis", "5"],
        ["   3.2 Mathematical Proofs", "6"],
        ["   3.3 Advanced Simulations", "7"],
        ["4. Game Theory Concepts Primer", "8"],
        ["5. Data Sources & Methodology", "9"],
        ["6. Troubleshooting & Support", "10"]
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
    This application is a PhD-level interactive research tool designed to analyze the structural transformation 
    of U.S.-China economic relations from 2001 to 2025 using rigorous game-theoretic frameworks. Unlike static 
    reports, this application allows users to manipulate variables, run simulations, and visually explore the 
    mathematical underpinnings of trade conflict.""", body_style))
    
    story.append(Paragraph("""
    <b>Key Objectives:</b><br/>
    • Model the shift from cooperative 'Harmony' to conflictual 'Prisoner's Dilemma'.<br/>
    • Quantify the impact of 'Vendor Financing' (Trade Deficit recycling) on Treasury yields.<br/>
    • Empirically validate 'Tit-for-Tat' tariff strategies during the 2018-2025 Trade War.<br/>
    • Provide dynamic visualizations of Nash Equilibria and Pareto Efficiency.
    """, body_style))

    # --- 2. NAVIGATION ---
    story.append(Paragraph("2. Navigation & User Interface", h2_style))
    story.append(Paragraph("The application uses a Sidebar Navigation menu on the left side of the screen. The main sections are:", body_style))

    nav_data = [
        ["Section", "Purpose", "Key Features"],
        ["Executive Summary", "High-level Findings", "Metric Cards, Key Correlations, Core Thesis"],
        ["Strategic Analysis", "Interactive Models", "Payoff Matrices, 3D Plots, Phase Diagrams"],
        ["Mathematical Proofs", "Formal Logic", "20+ Theorems, Derivations, Interactive Sliders"],
        ["Advanced Simulations", "Agent-Based Models", "Tournament Arena, Evolutionary Lab"],
        ["Methodology", "Academic Basis", "Data Sources, Assumptions, Citations"],
        ["Research Documents", "Library", "Access to PDF Reports, User Guide"]
    ]
    t_nav = Table(nav_data, colWidths=[1.5*inch, 1.5*inch, 3.5*inch])
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
    story.append(PageBreak())

    # --- 3. MODULES ---
    story.append(Paragraph("3. Core Functional Modules", h2_style))

    # 3.1 Strategic Analysis
    story.append(Paragraph("3.1 Strategic Analysis Module", h3_style))
    story.append(Paragraph("""
    This module visualizes the 2x2 Normal-Form Game between the U.S. and China.
    """, body_style))
    story.append(Paragraph("<b>Features:</b>", body_style))
    features_list = ListFlowable([
        ListItem(Paragraph("<b>Payoff Matrix Viewer:</b> Displays the 2x2 matrix (Cooperate/Defect) with updated payoff values based on the year selected.", body_style)),
        ListItem(Paragraph("<b>Nash Equilibrium Solver:</b> Automatically highlights the Nash Equilibrium (NE) and checks for Dominant Strategies.", body_style)),
        ListItem(Paragraph("<b>Pareto Efficiency Frontier:</b> Plots the payoffs to show if the NE is Pareto Efficient or if a better outcome exists (Prisoner's Dilemma).", body_style))
    ], bulletType='bullet', start='circle')
    story.append(features_list)

    # 3.2 Mathematical Proofs
    story.append(Paragraph("3.2 Mathematical Proofs Engine", h3_style))
    story.append(Paragraph("""
    Access formal derivations for over 20 theorems. Select a proof type from the sidebar dropdown.
    """, body_style))
    
    proof_data = [
        ["Category", "Examples"],
        ["Nash Equilibrium", "Existence, Uniqueness, Mixed Strategy"],
        ["Dominant Strategies", "Strict Dominance, Weak Dominance"],
        ["Repeated Games", "Folk Theorem, Grim Trigger, Tit-for-Tat"],
        ["Economic/Empirical", "Yield Suppression, Tariff Correlations"]
    ]
    t_proof = Table(proof_data, colWidths=[2*inch, 4.5*inch])
    t_proof.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e2e8f0')),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t_proof)

    # 3.3 Simulations
    story.append(Paragraph("3.3 Advanced Simulations Hub", h3_style))
    story.append(Paragraph("""
    <b>Tournament Arena:</b> Runs an Axelrod-style round-robin tournament. You can pit strategies like 'TitForTat', 'AlwaysDefect', and 'GrimTrigger' against each other.
    <br/><br/>
    <b>Evolutionary Lab:</b> Simulates a population of agents. Uses Replicator Dynamics to show how the population share of each strategy provides over generations. 
    <i>(Tip: Cooperation survives only if the Discount Factor δ is high enough.)</i>
    """, body_style))
    story.append(PageBreak())

    # --- 4. GAME THEORY CONCEPTS ---
    story.append(Paragraph("4. Game Theory Concepts Primer", h2_style))
    
    story.append(Paragraph("<b>The Prisoner's Dilemma</b>", h3_style))
    story.append(Paragraph("""
    The core model of the 2018-2025 Trade War. Even though Mutual Cooperation (Free Trade) is globally optimal, 
    Mutual Defection (Tariff War) is the unique Nash Equilibrium because each player has an incentive to defect.
    """, body_style))

    # Draw Matrix
    matrix_data = [
        ['', 'China Cooperate', 'China Defect'],
        ['US Cooperate', 'Reward (3, 3)\n(Free Trade)', 'Sucker (0, 5)\n(US Surplus)'],
        ['US Defect', 'Temptation (5, 0)\n(China Surplus)', 'Punishment (1, 1)\n(Trade War)']
    ]
    t_matrix = Table(matrix_data, colWidths=[1.5*inch, 2*inch, 2*inch])
    t_matrix.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BACKGROUND', (1,1), (1,1), colors.lightgreen), # Reward
        ('BACKGROUND', (2,2), (2,2), colors.salmon),    # Punishment (NE)
        ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
        ('FONTSIZE', (0,0), (-1,-1), 10),
        ('MINIMUMHEIGHT', (0,0), (-1,-1), 40)
    ]))
    story.append(t_matrix)
    story.append(Paragraph("<i>Fig 1: Standard Prisoner's Dilemma Payoff Matrix (Ordinal Payoffs)</i>", ParagraphStyle('Caption', parent=body_style, fontSize=9, alignment=TA_CENTER)))

    story.append(Paragraph("<b>Grim Trigger Strategy</b>", h3_style))
    story.append(Paragraph("""
    A strategy where a player cooperates until the opponent defects once, after which they defect forever. 
    It supports cooperation in repeated games if players are patient (high discount factor).
    """, body_style))

    # --- 5. DATA SOURCES ---
    story.append(Paragraph("5. Data Sources & Methodology", h2_style))
    story.append(Paragraph("All data is sourced from reputable government and international institutions.", body_style))
    
    data_sources = [
        ["Variable", "Source", "Frequency"],
        ["U.S. Trade Deficit", "U.S. Census Bureau", "Annual (2001-2025)"],
        ["China FX Reserves", "SAFE (State Admin of Foreign Exchange)", "Annual"],
        ["Treasury Yields (10Y)", "FRED (Federal Reserve)", "Daily/Annual Avg"],
        ["GDP Growth", "World Bank", "Annual"],
        ["Tariff Rates", "Peterson Institute (PIIE)", "Specific Events"]
    ]
    t_data = Table(data_sources, colWidths=[2*inch, 3*inch, 1.5*inch])
    t_data.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c5282')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))
    story.append(t_data)
    story.append(Spacer(1, 20))

    # --- 6. TROUBLESHOOTING ---
    story.append(Paragraph("6. Troubleshooting", h2_style))
    story.append(Paragraph("""
    • <b>Visualizations Blank:</b> Refresh the page. High interactive load can sometimes stall Plotly charts.<br/>
    • <b>PDFs Not Opening:</b> Ensure your browser allows pop-ups or use the embedded viewer.
    """, body_style))

    doc.build(story)
    print("User Guide.pdf created successfully.")

if __name__ == "__main__":
    create_comprehensive_guide()
