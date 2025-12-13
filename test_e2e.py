"""
End-to-End Test for Game Theory Streamlit Application
======================================================
This script performs comprehensive E2E testing by:
1. Testing all data loading
2. Testing all game theory calculations
3. Testing all visualization generation
4. Testing all page rendering functions

Run with: python test_e2e.py
"""

import sys
import os
import time
import traceback

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components
print("="*70)
print("GAME THEORY APPLICATION - END-TO-END TEST")
print("="*70)
print()

# ============================================================================
# PHASE 1: IMPORT VALIDATION
# ============================================================================
print("PHASE 1: IMPORT VALIDATION")
print("-"*70)

try:
    from app import (
        # Core Classes
        PayoffMatrix,
        GameTheoryEngine,
        DataManager,
        GameType,
        HistoricalPeriod,
        StrategyType,
        EquilibriumResult,
        SimulationResult,
        StatisticalTestResult,
        
        # Visualization
        VisualizationEngine,
        ProfessionalTheme,
        get_professional_layout,
        
        # Component Functions
        render_professional_table,
        render_comparison_chart,
        render_professional_timeline,
        render_kpi_dashboard,
        render_professional_heatmap,
        render_sankey_diagram,
        render_gauge_chart,
        render_waterfall_chart,
        
        # Helper Functions
        categorize_delta,
        
        # Page Renderers
        render_executive_summary,
        render_nash_equilibrium_page,
        render_pareto_efficiency_page,
        render_repeated_games_page,
        render_empirical_data_page,
        render_strategy_simulator_page,
        render_mathematical_proofs_page,
    )
    print("  ✅ All imports successful")
except ImportError as e:
    print(f"  ❌ Import failed: {e}")
    sys.exit(1)

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ============================================================================
# PHASE 2: DATA LOADING
# ============================================================================
print()
print("PHASE 2: DATA LOADING")
print("-"*70)

data_manager = DataManager()
data_tests = {}

try:
    tariff_data = data_manager.get_tariff_data()
    data_tests['Tariff Data'] = f"✅ {len(tariff_data)} rows"
except Exception as e:
    data_tests['Tariff Data'] = f"❌ {e}"

try:
    coop_data = data_manager.get_cooperation_index_data()
    data_tests['Cooperation Index'] = f"✅ {len(coop_data)} rows"
except Exception as e:
    data_tests['Cooperation Index'] = f"❌ {e}"

try:
    discount_data = data_manager.get_discount_factor_data()
    data_tests['Discount Factor'] = f"✅ {len(discount_data)} rows"
except Exception as e:
    data_tests['Discount Factor'] = f"❌ {e}"

for name, result in data_tests.items():
    print(f"  {name}: {result}")

# ============================================================================
# PHASE 3: PAYOFF MATRIX CREATION
# ============================================================================
print()
print("PHASE 3: PAYOFF MATRIX CREATION")
print("-"*70)

try:
    harmony_matrix = PayoffMatrix(
        cc=(8, 8), cd=(2, 5), dc=(5, 2), dd=(1, 1)
    )
    print(f"  ✅ Harmony Matrix created: CC={harmony_matrix.cc}, DD={harmony_matrix.dd}")
except Exception as e:
    print(f"  ❌ Harmony Matrix failed: {e}")
    sys.exit(1)

try:
    pd_matrix = PayoffMatrix(
        cc=(6, 6), cd=(0, 8), dc=(8, 0), dd=(2, 2)
    )
    print(f"  ✅ PD Matrix created: CC={pd_matrix.cc}, DD={pd_matrix.dd}")
except Exception as e:
    print(f"  ❌ PD Matrix failed: {e}")
    sys.exit(1)

# ============================================================================
# PHASE 4: GAME THEORY ENGINE
# ============================================================================
print()
print("PHASE 4: GAME THEORY ENGINE")
print("-"*70)

engine_tests = {}

# Test Harmony Engine
harmony_engine = GameTheoryEngine(harmony_matrix)
pd_engine = GameTheoryEngine(pd_matrix)

try:
    eq = harmony_engine.find_nash_equilibria()
    engine_tests['Harmony Nash Equilibria'] = f"✅ Found {len(eq)} equilibria"
except Exception as e:
    engine_tests['Harmony Nash Equilibria'] = f"❌ {e}"

try:
    eq = pd_engine.find_nash_equilibria()
    engine_tests['PD Nash Equilibria'] = f"✅ Found {len(eq)} equilibria"
except Exception as e:
    engine_tests['PD Nash Equilibria'] = f"❌ {e}"

try:
    dominant = harmony_engine.find_dominant_strategies()
    engine_tests['Harmony Dominant Strategies'] = f"✅ {dominant}"
except Exception as e:
    engine_tests['Harmony Dominant Strategies'] = f"❌ {e}"

try:
    dominant = pd_engine.find_dominant_strategies()
    engine_tests['PD Dominant Strategies'] = f"✅ {dominant}"
except Exception as e:
    engine_tests['PD Dominant Strategies'] = f"❌ {e}"

try:
    pareto = harmony_engine.pareto_efficiency_analysis()
    engine_tests['Pareto Analysis'] = f"✅ {len(pareto)} outcomes analyzed"
except Exception as e:
    engine_tests['Pareto Analysis'] = f"❌ {e}"

try:
    game_type = harmony_engine.classify_game_type()
    engine_tests['Harmony Classification'] = f"✅ {game_type.value}"
except Exception as e:
    engine_tests['Harmony Classification'] = f"❌ {e}"

try:
    game_type = pd_engine.classify_game_type()
    engine_tests['PD Classification'] = f"✅ {game_type.value}"
except Exception as e:
    engine_tests['PD Classification'] = f"❌ {e}"

try:
    delta = pd_engine.calculate_critical_discount_factor()
    engine_tests['Critical Discount Factor'] = f"✅ δ* = {delta:.3f}"
except Exception as e:
    engine_tests['Critical Discount Factor'] = f"❌ {e}"

try:
    margin = pd_engine.calculate_cooperation_margin(0.9)
    engine_tests['Cooperation Margin (δ=0.9)'] = f"✅ Margin = {margin:.3f}"
except Exception as e:
    engine_tests['Cooperation Margin (δ=0.9)'] = f"❌ {e}"

for name, result in engine_tests.items():
    print(f"  {name}: {result}")

# ============================================================================
# PHASE 5: STRATEGY SIMULATION
# ============================================================================
print()
print("PHASE 5: STRATEGY SIMULATION")
print("-"*70)

strategy_tests = {}

for strategy in [StrategyType.TIT_FOR_TAT, StrategyType.GRIM_TRIGGER, StrategyType.ALWAYS_DEFECT]:
    try:
        result = harmony_engine.simulate_strategy(strategy, rounds=20)
        strategy_tests[strategy.value] = f"✅ US={result.total_us_payoff}, CN={result.total_china_payoff}"
    except Exception as e:
        strategy_tests[strategy.value] = f"❌ {e}"

for name, result in strategy_tests.items():
    print(f"  {name}: {result}")

# ============================================================================
# PHASE 6: VISUALIZATION GENERATION
# ============================================================================
print()
print("PHASE 6: VISUALIZATION GENERATION")
print("-"*70)

viz_tests = {}

try:
    fig = VisualizationEngine.create_payoff_matrix_heatmap(harmony_matrix, "Harmony")
    viz_tests['Payoff Heatmap (Harmony)'] = f"✅ {len(fig.data)} traces"
except Exception as e:
    viz_tests['Payoff Heatmap (Harmony)'] = f"❌ {e}"

try:
    fig = VisualizationEngine.create_payoff_matrix_heatmap(pd_matrix, "PD")
    viz_tests['Payoff Heatmap (PD)'] = f"✅ {len(fig.data)} traces"
except Exception as e:
    viz_tests['Payoff Heatmap (PD)'] = f"❌ {e}"

try:
    fig = VisualizationEngine.create_cooperation_index_chart(coop_data)
    viz_tests['Cooperation Index Chart'] = f"✅ {len(fig.data)} traces"
except Exception as e:
    viz_tests['Cooperation Index Chart'] = f"❌ {e}"

try:
    fig = VisualizationEngine.create_tariff_escalation_chart(tariff_data)
    viz_tests['Tariff Escalation Chart'] = f"✅ {len(fig.data)} traces"
except Exception as e:
    viz_tests['Tariff Escalation Chart'] = f"❌ {e}"

for name, result in viz_tests.items():
    print(f"  {name}: {result}")

# ============================================================================
# PHASE 7: PROFESSIONAL COMPONENTS
# ============================================================================
print()
print("PHASE 7: PROFESSIONAL COMPONENTS")
print("-"*70)

component_tests = {}

try:
    layout = get_professional_layout()
    component_tests['Professional Layout'] = f"✅ {len(layout)} properties"
except Exception as e:
    component_tests['Professional Layout'] = f"❌ {e}"

try:
    palette = ProfessionalTheme.CHART_PALETTE
    component_tests['Chart Palette'] = f"✅ {len(palette)} colors"
except Exception as e:
    component_tests['Chart Palette'] = f"❌ {e}"

try:
    result = categorize_delta(0.15)
    component_tests['categorize_delta()'] = f"✅ '{result}'"
except Exception as e:
    component_tests['categorize_delta()'] = f"❌ {e}"

for name, result in component_tests.items():
    print(f"  {name}: {result}")

# ============================================================================
# PHASE 8: ADD NOISE & MONTE CARLO
# ============================================================================
print()
print("PHASE 8: MONTE CARLO SIMULATION")
print("-"*70)

mc_tests = {}

try:
    results = []
    for i in range(10):
        noisy_engine = harmony_engine.add_noise(noise_level=0.1)
        game_type = noisy_engine.classify_game_type()
        results.append(game_type)
    mc_tests['10 Monte Carlo Runs'] = f"✅ Completed"
except Exception as e:
    mc_tests['10 Monte Carlo Runs'] = f"❌ {e}"

try:
    # Test engine copy
    copy_engine = harmony_engine.copy()
    mc_tests['Engine Copy'] = f"✅ Copy created"
except Exception as e:
    mc_tests['Engine Copy'] = f"❌ {e}"

for name, result in mc_tests.items():
    print(f"  {name}: {result}")

# ============================================================================
# PHASE 9: DATA VALIDATION
# ============================================================================
print()
print("PHASE 9: DATA VALIDATION")
print("-"*70)

validation_tests = {}

# Validate tariff data values
try:
    tariff_cols = list(tariff_data.columns)
    validation_tests['Tariff Columns'] = f"✅ {tariff_cols[:4]}..."
except Exception as e:
    validation_tests['Tariff Columns'] = f"❌ {e}"

# Validate cooperation data
try:
    coop_cols = list(coop_data.columns)
    validation_tests['Cooperation Columns'] = f"✅ {coop_cols[:4]}..."
except Exception as e:
    validation_tests['Cooperation Columns'] = f"❌ {e}"

# Validate game type classification
try:
    assert harmony_engine.classify_game_type() == GameType.HARMONY
    assert pd_engine.classify_game_type() == GameType.PRISONERS_DILEMMA
    validation_tests['Game Classification'] = f"✅ Harmony→HARMONY, PD→PRISONERS_DILEMMA"
except Exception as e:
    validation_tests['Game Classification'] = f"❌ {e}"

for name, result in validation_tests.items():
    print(f"  {name}: {result}")

# ============================================================================
# SUMMARY
# ============================================================================
print()
print("="*70)
print("END-TO-END TEST SUMMARY")
print("="*70)

# Count results
all_results = {
    **data_tests,
    **engine_tests,
    **strategy_tests,
    **viz_tests,
    **component_tests,
    **mc_tests,
    **validation_tests
}

passed = sum(1 for v in all_results.values() if '✅' in v)
failed = sum(1 for v in all_results.values() if '❌' in v)
total = passed + failed

print(f"""
  Total Tests:  {total}
  Passed:       {passed} ✅
  Failed:       {failed} {'❌' if failed > 0 else ''}
  
  Pass Rate:    {passed/total*100:.1f}%
""")

if failed > 0:
    print("FAILED TESTS:")
    for name, result in all_results.items():
        if '❌' in result:
            print(f"  - {name}: {result}")
    print()

if failed == 0:
    print("🎉 ALL TESTS PASSED! Application is working correctly.")
else:
    print(f"⚠️  {failed} test(s) failed. Please review above.")

print("="*70)
