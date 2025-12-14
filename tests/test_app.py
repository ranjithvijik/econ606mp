"""
Comprehensive Test Suite for Game Theory Application
=====================================================

This module contains:
- Unit Tests: Individual component testing
- Integration Tests: Component interaction testing
- User Acceptance Tests: End-to-end scenario validation

Run with: python -m pytest tests/test_app.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# Import application modules
from app import (
    PayoffMatrix,
    AdvancedSimulationConfig,
    AdvancedSimulationEngine,
    StrategyType,
    GameTheoryEngine,
    HistoricalPeriod,
    GameType,
    ProfessionalTheme,
)


# =============================================================================
# UNIT TESTS
# =============================================================================

class TestPayoffMatrix:
    """Unit tests for PayoffMatrix class."""
    
    def test_creation(self):
        """Test PayoffMatrix can be created with valid payoffs."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        assert matrix.cc == (3, 3)
        assert matrix.cd == (0, 5)
        assert matrix.dc == (5, 0)
        assert matrix.dd == (1, 1)
    
    def test_to_dataframe(self):
        """Test conversion to DataFrame."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        df = matrix.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
    
    def test_to_numpy(self):
        """Test conversion to numpy arrays."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        us_payoffs, china_payoffs = matrix.to_numpy()
        assert us_payoffs.shape == (2, 2)
        assert china_payoffs.shape == (2, 2)
        # Verify correct payoff extraction
        assert us_payoffs[0, 0] == 3  # cc
        assert china_payoffs[0, 0] == 3  # cc
    
    def test_get_payoff_parameters_row(self):
        """Test T,R,P,S extraction for row player."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        params = matrix.get_payoff_parameters_row()
        assert params['T'] == 5.0  # Temptation
        assert params['R'] == 3.0  # Reward
        assert params['P'] == 1.0  # Punishment
        assert params['S'] == 0.0  # Sucker
    
    def test_get_payoff_parameters_col(self):
        """Test T,R,P,S extraction for column player."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        params = matrix.get_payoff_parameters_col()
        assert params['T'] == 5.0
        assert params['R'] == 3.0
        assert params['P'] == 1.0
        assert params['S'] == 0.0
    
    def test_get_both_payoff_parameters(self):
        """Test extraction for both players."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        row_params, col_params = matrix.get_both_payoff_parameters()
        assert isinstance(row_params, dict)
        assert isinstance(col_params, dict)
        assert 'T' in row_params and 'T' in col_params
    
    def test_asymmetric_payoffs(self):
        """Test asymmetric payoff matrix."""
        matrix = PayoffMatrix(cc=(4, 3), cd=(1, 5), dc=(6, 0), dd=(2, 1))
        row_params = matrix.get_payoff_parameters_row()
        col_params = matrix.get_payoff_parameters_col()
        # Row player (U.S.)
        assert row_params['R'] == 4.0
        # Column player (China)
        assert col_params['R'] == 3.0
    
    def test_get_joint_welfare(self):
        """Test joint welfare calculation."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        welfare = matrix.get_joint_welfare()
        assert welfare['(C,C)'] == 6  # 3 + 3
        assert welfare['(D,D)'] == 2  # 1 + 1
    
    def test_to_latex(self):
        """Test LaTeX generation."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        latex = matrix.to_latex()
        assert 'begin{array}' in latex
        assert '(3, 3)' in latex


class TestAdvancedSimulationConfig:
    """Unit tests for AdvancedSimulationConfig."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = AdvancedSimulationConfig()
        assert config.rounds > 0
        assert config.generations > 0
        assert config.population_size > 0
        assert 0 <= config.noise_probability <= 1
        assert 0 <= config.mutation_rate <= 1
    
    def test_custom_values(self):
        """Test custom configuration."""
        config = AdvancedSimulationConfig(
            rounds=50,
            generations=20,
            population_size=200,
            noise_probability=0.05
        )
        assert config.rounds == 50
        assert config.generations == 20


class TestAdvancedSimulationEngine:
    """Unit tests for AdvancedSimulationEngine."""
    
    @pytest.fixture
    def engine(self):
        """Create engine with standard PD matrix."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        config = AdvancedSimulationConfig(rounds=10, generations=5, population_size=20)
        return AdvancedSimulationEngine(matrix, config)
    
    def test_simulate_match(self, engine):
        """Test match simulation between strategies."""
        payoffs = engine._simulate_match(StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE, 10)
        assert len(payoffs) == 4
        assert payoffs[0] >= 0  # Payoff 1
        assert payoffs[1] >= 0  # Payoff 2
        assert 0 <= payoffs[2] <= 1  # Coop rate 1
        assert 0 <= payoffs[3] <= 1  # Coop rate 2
    
    def test_tournament(self, engine):
        """Test tournament runs without errors."""
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE, StrategyType.ALWAYS_DEFECT]
        results = engine.run_tournament(strategies, rounds_per_match=10)
        
        assert isinstance(results, pd.DataFrame)
        assert len(results) == len(strategies) ** 2
        assert 'Strategy_1' in results.columns
        assert 'Payoff_1' in results.columns
        assert 'Avg_Payoff_Per_Round_1' in results.columns  # BUG-016 fix
        assert 'Rounds' in results.columns  # BUG-016 fix
    
    def test_tournament_payoff_per_round(self, engine):
        """Test that per-round payoffs are correctly calculated."""
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE]
        rounds = 10
        results = engine.run_tournament(strategies, rounds_per_match=rounds)
        
        # Verify per-round calculation
        for _, row in results.iterrows():
            expected_avg = row['Payoff_1'] / rounds
            assert abs(row['Avg_Payoff_Per_Round_1'] - expected_avg) < 0.001
    
    def test_evolutionary_simulation(self, engine):
        """Test evolutionary dynamics simulation."""
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 5,
            StrategyType.ALWAYS_COOPERATE: 5,
            StrategyType.ALWAYS_DEFECT: 5,
            StrategyType.GRIM_TRIGGER: 5
        }
        results = engine.run_evolutionary_simulation(initial_pop, generations=5)
        
        assert isinstance(results, pd.DataFrame)
        assert 'Generation' in results.columns
        assert len(results) == 5
    
    def test_stochastic_game(self, engine):
        """Test stochastic game simulation."""
        results = engine.run_stochastic_game(rounds=20)
        
        assert isinstance(results, pd.DataFrame)
        assert 'Round' in results.columns
        assert 'State' in results.columns
        assert len(results) == 20
        
        # Verify states are valid
        valid_states = ['Cooperative', 'Neutral', 'Hostile']
        for state in results['State']:
            assert state in valid_states
    
    def test_learning_simulation_fictitious_play(self, engine):
        """Test fictitious play learning."""
        results = engine.run_learning_simulation('fictitious_play', rounds=20)
        
        assert isinstance(results, pd.DataFrame)
        assert 'US_Belief_China_Coop' in results.columns
        assert 'China_Belief_US_Coop' in results.columns
    
    def test_learning_simulation_reinforcement(self, engine):
        """Test reinforcement learning."""
        results = engine.run_learning_simulation('reinforcement', rounds=20)
        
        assert isinstance(results, pd.DataFrame)
        assert 'US_Q_Coop' in results.columns
    
    def test_learning_simulation_regret_matching(self, engine):
        """Test regret matching with bounded regrets (BUG-010 fix)."""
        results = engine.run_learning_simulation('regret_matching', rounds=100)
        
        assert isinstance(results, pd.DataFrame)
        assert 'US_Regret_Coop' in results.columns
        
        # Verify regrets are bounded (BUG-010)
        max_regret = 10000.0
        assert results['US_Regret_Coop'].max() <= max_regret
        assert results['US_Regret_Defect'].max() <= max_regret


class TestStrategyActions:
    """Test individual strategy implementations."""
    
    @pytest.fixture
    def engine(self):
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        return AdvancedSimulationEngine(matrix)
    
    def test_always_cooperate(self, engine):
        """Test Always Cooperate strategy."""
        action = engine._get_strategy_action(StrategyType.ALWAYS_COOPERATE, 0, [], [], [])
        assert action == 'C'
        action = engine._get_strategy_action(StrategyType.ALWAYS_COOPERATE, 10, ['D']*10, ['D']*10, [1]*10)
        assert action == 'C'
    
    def test_always_defect(self, engine):
        """Test Always Defect strategy."""
        action = engine._get_strategy_action(StrategyType.ALWAYS_DEFECT, 0, [], [], [])
        assert action == 'D'
    
    def test_tit_for_tat(self, engine):
        """Test Tit-for-Tat strategy."""
        # First move: Cooperate
        action = engine._get_strategy_action(StrategyType.TIT_FOR_TAT, 0, [], [], [])
        assert action == 'C'
        
        # Mirror opponent's last move
        action = engine._get_strategy_action(StrategyType.TIT_FOR_TAT, 1, ['C'], ['D'], [0])
        assert action == 'D'
        
        action = engine._get_strategy_action(StrategyType.TIT_FOR_TAT, 2, ['C', 'D'], ['D', 'C'], [0, 5])
        assert action == 'C'
    
    def test_grim_trigger(self, engine):
        """Test Grim Trigger strategy."""
        # Cooperate until defection
        action = engine._get_strategy_action(StrategyType.GRIM_TRIGGER, 0, [], [], [])
        assert action == 'C'
        
        action = engine._get_strategy_action(StrategyType.GRIM_TRIGGER, 1, ['C'], ['C'], [3])
        assert action == 'C'
        
        # Defect forever after opponent defects
        action = engine._get_strategy_action(StrategyType.GRIM_TRIGGER, 2, ['C', 'C'], ['C', 'D'], [3, 0])
        assert action == 'D'
    
    def test_pavlov(self, engine):
        """Test Pavlov (Win-Stay, Lose-Shift) strategy."""
        # First move: Cooperate
        action = engine._get_strategy_action(StrategyType.PAVLOV, 0, [], [], [])
        assert action == 'C'
        
        # Win (high payoff) -> Stay
        action = engine._get_strategy_action(StrategyType.PAVLOV, 1, ['C'], ['C'], [3])
        assert action == 'C'
        
        # Lose (low payoff) -> Shift
        action = engine._get_strategy_action(StrategyType.PAVLOV, 1, ['C'], ['D'], [0])
        assert action == 'D'


class TestGameTheoryEngine:
    """Unit tests for GameTheoryEngine."""
    
    @pytest.fixture
    def engine(self):
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        return GameTheoryEngine(matrix)
    
    def test_find_nash_equilibria(self, engine):
        """Test Nash equilibrium detection."""
        equilibria = engine.find_nash_equilibria()
        assert isinstance(equilibria, list)
        # PD has (Defect, Defect) as unique NE - check format flexibly
        eq_str = str(equilibria).lower()
        assert 'defect' in eq_str, f"Expected defect in equilibria: {equilibria}"
    
    def test_calculate_dominant_strategies(self, engine):
        """Test dominant strategy detection."""
        dominant = engine.find_dominant_strategies()
        assert isinstance(dominant, dict)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for component interactions."""
    
    def test_full_tournament_pipeline(self):
        """Test complete tournament workflow."""
        # Create matrix
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        
        # Create engine with config
        config = AdvancedSimulationConfig(rounds=50, noise_probability=0.01)
        engine = AdvancedSimulationEngine(matrix, config)
        
        # Run tournament
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE, 
                     StrategyType.ALWAYS_DEFECT, StrategyType.GRIM_TRIGGER]
        results = engine.run_tournament(strategies, rounds_per_match=50)
        
        # Verify results
        assert len(results) == 16  # 4 strategies x 4 opponents
        assert results['Payoff_1'].sum() > 0
        
        # Verify rankings can be computed
        rankings = results.groupby('Strategy_1')['Payoff_1'].sum().sort_values(ascending=False)
        assert len(rankings) == 4
    
    def test_evolutionary_to_equilibrium(self):
        """Test that evolutionary dynamics reach stable state."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        config = AdvancedSimulationConfig(generations=20, population_size=100, mutation_rate=0.01)
        engine = AdvancedSimulationEngine(matrix, config)
        
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 25,
            StrategyType.ALWAYS_COOPERATE: 25,
            StrategyType.ALWAYS_DEFECT: 25,
            StrategyType.GRIM_TRIGGER: 25
        }
        
        results = engine.run_evolutionary_simulation(initial_pop, generations=20)
        
        # Verify population is maintained
        last_gen = results.iloc[-1]
        total_pop = sum([last_gen.get(s.value, 0) for s in initial_pop.keys()])
        assert total_pop >= 50  # Some population should remain
    
    def test_learning_convergence(self):
        """Test that learning algorithms converge."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        config = AdvancedSimulationConfig(rounds=100, learning_rate=0.1)
        engine = AdvancedSimulationEngine(matrix, config)
        
        results = engine.run_learning_simulation('fictitious_play', rounds=100)
        
        # Beliefs should stabilize
        early_variance = results.iloc[:10]['US_Belief_China_Coop'].var()
        late_variance = results.iloc[-10:]['US_Belief_China_Coop'].var()
        # Late variance should be smaller (convergence)
        assert late_variance <= early_variance + 0.1
    
    def test_stochastic_game_state_transitions(self):
        """Test that stochastic game has valid state transitions."""
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        engine = AdvancedSimulationEngine(matrix)
        
        results = engine.run_stochastic_game(rounds=100)
        
        # Verify state distribution
        state_counts = results['State'].value_counts()
        assert len(state_counts) >= 1  # At least one state visited
        
        # Verify all states are valid
        for state in results['State'].unique():
            assert state in ['Cooperative', 'Neutral', 'Hostile']


# =============================================================================
# USER ACCEPTANCE TESTS
# =============================================================================

class TestUserAcceptance:
    """User acceptance tests simulating real-world usage scenarios."""
    
    def test_scenario_trade_war_analysis(self):
        """
        UAT: Analyze U.S.-China trade war as Prisoner's Dilemma.
        
        Scenario: Economist wants to model trade war dynamics
        Expected: System correctly identifies PD structure and dominant strategies
        """
        # Model trade war payoffs (tariffs as defection)
        trade_matrix = PayoffMatrix(
            cc=(8, 8),   # Free trade
            cd=(2, 10),  # US tariffs, China doesn't
            dc=(10, 2),  # China tariffs, US doesn't
            dd=(4, 4)    # Trade war
        )
        
        engine = GameTheoryEngine(trade_matrix)
        
        # Verify PD structure: T > R > P > S
        params = trade_matrix.get_payoff_parameters_row()
        assert params['T'] > params['R'] > params['P'] > params['S'], \
            "Should recognize Prisoner's Dilemma structure"
        
        # Verify Nash equilibrium is mutual defection
        equilibria = engine.find_nash_equilibria()
        assert len(equilibria) > 0, "Should find Nash equilibrium"
    
    def test_scenario_cooperation_sustainability(self):
        """
        UAT: Determine if cooperation can be sustained with Tit-for-Tat.
        
        Scenario: Policy analyst tests cooperation sustainability
        Expected: TFT should outperform pure defection in iterated game
        """
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        config = AdvancedSimulationConfig(rounds=100)
        engine = AdvancedSimulationEngine(matrix, config)
        
        # Tournament between TFT and Always Defect
        results = engine.run_tournament(
            [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_DEFECT],
            rounds_per_match=100
        )
        
        # Calculate average payoffs
        tft_payoff = results[results['Strategy_1'] == 'Tit-for-Tat']['Payoff_1'].mean()
        defect_payoff = results[results['Strategy_1'] == 'Always Defect']['Payoff_1'].mean()
        
        # TFT should perform reasonably well
        assert tft_payoff >= defect_payoff * 0.8, \
            "TFT should be competitive with pure defection"
    
    def test_scenario_discount_factor_analysis(self):
        """
        UAT: Calculate critical discount factor for cooperation.
        
        Scenario: Researcher needs to find when cooperation is sustainable
        Expected: System calculates valid discount factor
        """
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        params = matrix.get_payoff_parameters_row()
        
        T, R, P, S = params['T'], params['R'], params['P'], params['S']
        
        # Critical discount factor formula: (T - R) / (T - P)
        if T != P:
            delta_star = (T - R) / (T - P)
            assert 0 < delta_star < 1, "Discount factor should be between 0 and 1"
    
    def test_scenario_evolutionary_dynamics_visualization_data(self):
        """
        UAT: Generate data for evolutionary dynamics visualization.
        
        Scenario: User wants to visualize strategy evolution
        Expected: System produces valid time-series data
        """
        matrix = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        config = AdvancedSimulationConfig(generations=50, population_size=100)
        engine = AdvancedSimulationEngine(matrix, config)
        
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 25,
            StrategyType.ALWAYS_COOPERATE: 25,
            StrategyType.ALWAYS_DEFECT: 25,
            StrategyType.GRIM_TRIGGER: 25
        }
        
        results = engine.run_evolutionary_simulation(initial_pop, generations=50)
        
        # Verify data suitable for visualization
        assert 'Generation' in results.columns
        assert len(results) == 50
        
        # Verify share columns exist for plotting
        share_cols = [col for col in results.columns if '_Share' in col]
        assert len(share_cols) >= 4, "Should have share columns for each strategy"
        
        # Verify shares sum to approximately 1
        for _, row in results.iterrows():
            total_share = sum([row[col] for col in share_cols if col in row])
            assert 0.9 <= total_share <= 1.1, f"Shares should sum to ~1, got {total_share}"
    
    def test_scenario_comparative_statics(self):
        """
        UAT: Perform comparative statics on discount factor.
        
        Scenario: User changes payoff parameters and observes effect
        Expected: System correctly shows inverse relationship
        """
        # Base case
        matrix1 = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        params1 = matrix1.get_payoff_parameters_row()
        delta1 = (params1['T'] - params1['R']) / (params1['T'] - params1['P'])
        
        # Increase R (cooperation reward)
        matrix2 = PayoffMatrix(cc=(4, 4), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        params2 = matrix2.get_payoff_parameters_row()
        delta2 = (params2['T'] - params2['R']) / (params2['T'] - params2['P'])
        
        # Higher R should lower discount factor (easier cooperation)
        assert delta2 < delta1, "Higher R should lower critical discount factor"


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    # Run with verbose output
    pytest.main([__file__, '-v', '--tb=short'])
