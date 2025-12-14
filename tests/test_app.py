import sys
import unittest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock

# =============================================================================
# MOCKING STREAMLIT ENVIRONMENT
# =============================================================================
# We must mock streamlit before importing app.py to avoid 
# executing st.set_page_config() and other UI interactions.
st_mock = MagicMock()
st_mock.session_state = {}
st_mock.sidebar = MagicMock()
# Simple pass-through for decorators
st_mock.cache_data = lambda func: func 
st_mock.cache_resource = lambda func: func
sys.modules['streamlit'] = st_mock

# Now we can import the app modules safely
# We assume app.py is in the parent or current directory
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import app

class TestCoreLogic(unittest.TestCase):
    """Unit tests for Game Theory Core Logic."""

    def setUp(self):
        # Standard Prisoner's Dilemma: T=5, R=3, P=1, S=0
        self.pd_matrix = app.PayoffMatrix(
            cc=(3.0, 3.0), 
            cd=(0.0, 5.0), 
            dc=(5.0, 0.0), 
            dd=(1.0, 1.0)
        )
        self.engine = app.GameTheoryEngine(self.pd_matrix)

    def test_payoff_matrix_validation(self):
        """Test PayoffMatrix inputs validation."""
        # Valid
        try:
            app.PayoffMatrix(cc=(3,3), cd=(0,5), dc=(5,0), dd=(1,1))
        except ValueError:
            self.fail("PayoffMatrix raised ValueError unexpectedly!")

        # Invalid dimensions
        with self.assertRaises(ValueError):
            app.PayoffMatrix(cc=(3,3,3), cd=(0,5), dc=(5,0), dd=(1,1))
            
        # Invalid types
        with self.assertRaises(ValueError):
            app.PayoffMatrix(cc=('a','b'), cd=(0,5), dc=(5,0), dd=(1,1))

    def test_game_classification(self):
        """Test correct classification of games."""
        # PD: T > R > P > S
        self.assertEqual(self.engine.classify_game_type(), app.GameType.PRISONERS_DILEMMA)
        
        # Harmony: R > T > S > P (6 > 5 > 2 > 1)
        harmony = app.PayoffMatrix(cc=(6,6), cd=(2,5), dc=(5,2), dd=(1,1))
        h_engine = app.GameTheoryEngine(harmony)
        self.assertEqual(h_engine.classify_game_type(), app.GameType.HARMONY)

    def test_nash_equilibria_pd(self):
        """Test Nash Equilibrium finding for PD."""
        nash = self.engine.find_nash_equilibria()
        self.assertIn("(Defect, Defect)", nash)
        self.assertEqual(len(nash), 1)

    def test_nash_equilibria_coordination(self):
        """Test Nash for Coordination/Stag Hunt."""
        # Stag Hunt: R > T > P > S (5 > 4 > 2 > 0)
        sh_matrix = app.PayoffMatrix(cc=(5,5), cd=(0,4), dc=(4,0), dd=(2,2))
        sh_engine = app.GameTheoryEngine(sh_matrix)
        nash = sh_engine.find_nash_equilibria()
        
        self.assertIn("(Cooperate, Cooperate)", nash)
        self.assertIn("(Defect, Defect)", nash)
        self.assertEqual(len(nash), 2)

    def test_critical_discount_factor(self):
        """Test Folk Theorem threshold calculation."""
        # δ* = (T - R) / (T - P) = (5 - 3) / (5 - 1) = 2/4 = 0.5
        crit_delta = self.engine.calculate_critical_discount_factor()
        self.assertAlmostEqual(crit_delta, 0.5)

class TestAdvancedSimulation(unittest.TestCase):
    """Integration tests for Advanced Simulations."""

    def setUp(self):
        self.pd_matrix = app.PayoffMatrix(
            cc=(3.0, 3.0), cd=(0.0, 5.0), dc=(5.0, 0.0), dd=(1.0, 1.0)
        )
        self.config = app.AdvancedSimulationConfig(rounds=20, population_size=10, generations=5)
        self.engine = app.AdvancedSimulationEngine(self.pd_matrix, self.config)

    def test_tournament_execution(self):
        """Test that tournament runs and returns valid DataFrame."""
        strategies = [
            app.StrategyType.TIT_FOR_TAT, 
            app.StrategyType.ALWAYS_DEFECT,
            app.StrategyType.ALWAYS_COOPERATE
        ]
        results = self.engine.run_tournament(strategies, rounds_per_match=10)
        
        self.assertIsInstance(results, pd.DataFrame)
        # 3 strategies, plays everyone (including self) = 3*3 = 9 matchups
        self.assertEqual(len(results), 9)
        self.assertTrue('Payoff_1' in results.columns)

    def test_evolutionary_dynamics(self):
        """Test evolutionary loop."""
        results = self.engine.run_evolutionary_simulation()
        self.assertIsInstance(results, pd.DataFrame)
        # Should have rows equal to generations
        self.assertEqual(len(results), self.config.generations)
        
    def test_spatial_simulation(self):
        """Test the new Spatial Simulation logic."""
        init_dist = {
            app.StrategyType.TIT_FOR_TAT: 0.5,
            app.StrategyType.ALWAYS_DEFECT: 0.5
        }
        grid_size = 10
        gens = 5
        
        history, strats = self.engine.run_spatial_simulation(grid_size, init_dist, gens)
        
        # Check history length (initial + gens)
        self.assertEqual(len(history), gens + 1)
        # Check grid shapes
        self.assertEqual(history[0].shape, (grid_size, grid_size))
        # Check strategies list
        self.assertEqual(len(strats), 2)

    def test_custom_strategy(self):
        """Test Gene Lab Custom Strategy Logic."""
        custom_params = {'p_CC': 1.0, 'p_CD': 0.0, 'p_DC': 0.0, 'p_DD': 0.0, 'p_init': 1.0}
        self.engine.config.custom_strategy_params = custom_params
        
        # Test Custom vs Cooperator
        # Round 1: Custom(C) vs Coop(C) -> Payoff R
        strat1 = app.StrategyType.CUSTOM
        strat2 = app.StrategyType.ALWAYS_COOPERATE
        
        payoff1, payoff2, _, _ = self.engine._simulate_match(strat1, strat2, rounds=5)
        
        # Custom here is effectively Tit-For-Tat (reciprocity 1.0, others 0.0)
        # It should cooperate fully with ALL_C
        expected_score = 3.0 * 5 # 15
        self.assertEqual(payoff1, expected_score)

if __name__ == '__main__':
    unittest.main()
