"""
Comprehensive Test Suite for Game Theory Application
=====================================================

This module provides exhaustive test coverage for all app functionality:
- Unit Tests: Individual component testing
- Integration Tests: Component interaction testing
- Functional Tests: Feature-level testing
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
    VisualizationEngine,
    AdvancedVisualizationEngine,
)


# =============================================================================
# FIXTURES - Reusable Test Components
# =============================================================================

@pytest.fixture
def pd_matrix():
    """Standard Prisoner's Dilemma payoff matrix."""
    return PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))

@pytest.fixture
def harmony_matrix():
    """Harmony Game payoff matrix (no conflict)."""
    return PayoffMatrix(cc=(4, 4), cd=(3, 2), dc=(2, 3), dd=(1, 1))

@pytest.fixture
def asymmetric_matrix():
    """Asymmetric payoff matrix for testing player-specific calculations."""
    return PayoffMatrix(cc=(4, 3), cd=(1, 5), dc=(6, 0), dd=(2, 1))

@pytest.fixture
def basic_config():
    """Basic simulation configuration."""
    return AdvancedSimulationConfig(rounds=10, generations=5, population_size=20)

@pytest.fixture
def engine(pd_matrix, basic_config):
    """Simulation engine with PD matrix and basic config."""
    return AdvancedSimulationEngine(pd_matrix, basic_config)

@pytest.fixture
def game_engine(pd_matrix):
    """Game theory engine for equilibrium analysis."""
    return GameTheoryEngine(pd_matrix)


# =============================================================================
# UNIT TESTS - PayoffMatrix
# =============================================================================

class TestPayoffMatrix:
    """Unit tests for PayoffMatrix data class."""
    
    def test_creation_valid(self, pd_matrix):
        """Test PayoffMatrix creation with valid payoffs."""
        assert pd_matrix.cc == (3, 3)
        assert pd_matrix.cd == (0, 5)
        assert pd_matrix.dc == (5, 0)
        assert pd_matrix.dd == (1, 1)
    
    def test_creation_different_types(self):
        """Test PayoffMatrix with different numeric types."""
        matrix = PayoffMatrix(cc=(3.5, 3.5), cd=(0.0, 5.0), dc=(5.0, 0.0), dd=(1.0, 1.0))
        assert matrix.cc[0] == 3.5
    
    def test_to_dataframe(self, pd_matrix):
        """Test conversion to pandas DataFrame."""
        df = pd_matrix.to_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)
        assert 'China' in df.columns[0]
    
    def test_to_numpy(self, pd_matrix):
        """Test conversion to numpy arrays."""
        us_payoffs, china_payoffs = pd_matrix.to_numpy()
        assert us_payoffs.shape == (2, 2)
        assert china_payoffs.shape == (2, 2)
        assert us_payoffs[0, 0] == 3
        assert china_payoffs[0, 0] == 3
        assert us_payoffs[0, 1] == 0  # cd for US
        assert china_payoffs[0, 1] == 5  # cd for China
    
    def test_get_payoff_parameters_row(self, pd_matrix):
        """Test T,R,P,S extraction for row player (US)."""
        params = pd_matrix.get_payoff_parameters_row()
        assert params['T'] == 5.0
        assert params['R'] == 3.0
        assert params['P'] == 1.0
        assert params['S'] == 0.0
    
    def test_get_payoff_parameters_col(self, pd_matrix):
        """Test T,R,P,S extraction for column player (China)."""
        params = pd_matrix.get_payoff_parameters_col()
        assert params['T'] == 5.0
        assert params['R'] == 3.0
        assert params['P'] == 1.0
        assert params['S'] == 0.0
    
    def test_asymmetric_payoff_parameters(self, asymmetric_matrix):
        """Test parameter extraction for asymmetric games."""
        row_params = asymmetric_matrix.get_payoff_parameters_row()
        col_params = asymmetric_matrix.get_payoff_parameters_col()
        
        # Row player (US) parameters
        assert row_params['R'] == 4.0  # cc[0]
        assert row_params['T'] == 6.0  # dc[0]
        assert row_params['S'] == 1.0  # cd[0]
        assert row_params['P'] == 2.0  # dd[0]
        
        # Column player (China) parameters should differ
        assert col_params['R'] == 3.0  # cc[1]
        assert col_params['T'] == 5.0  # cd[1]
    
    def test_get_both_payoff_parameters(self, pd_matrix):
        """Test extraction for both players."""
        row_params, col_params = pd_matrix.get_both_payoff_parameters()
        assert row_params == col_params  # Symmetric game
    
    def test_get_joint_welfare(self, pd_matrix):
        """Test joint welfare calculation."""
        welfare = pd_matrix.get_joint_welfare()
        assert welfare['(C,C)'] == 6
        assert welfare['(C,D)'] == 5
        assert welfare['(D,C)'] == 5
        assert welfare['(D,D)'] == 2
    
    def test_to_latex(self, pd_matrix):
        """Test LaTeX generation."""
        latex = pd_matrix.to_latex()
        assert 'begin{array}' in latex
        assert '(3, 3)' in latex
        assert 'China' in latex
    
    def test_prisoners_dilemma_structure(self, pd_matrix):
        """Verify payoff ordering satisfies PD: T > R > P > S."""
        params = pd_matrix.get_payoff_parameters_row()
        assert params['T'] > params['R'] > params['P'] > params['S']


# =============================================================================
# UNIT TESTS - AdvancedSimulationConfig
# =============================================================================

class TestAdvancedSimulationConfig:
    """Unit tests for simulation configuration."""
    
    def test_default_values(self):
        """Test default configuration values are reasonable."""
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
            noise_probability=0.05,
            mutation_rate=0.02
        )
        assert config.rounds == 50
        assert config.generations == 20
        assert config.population_size == 200


# =============================================================================
# UNIT TESTS - AdvancedSimulationEngine
# =============================================================================

class TestAdvancedSimulationEngine:
    """Unit tests for simulation engine."""
    
    def test_initialization(self, engine):
        """Test engine initialization."""
        assert engine.matrix is not None
        assert engine.config is not None
    
    def test_simulate_match_returns_correct_shape(self, engine):
        """Test match simulation returns 4 values."""
        result = engine._simulate_match(
            StrategyType.TIT_FOR_TAT, 
            StrategyType.ALWAYS_COOPERATE, 
            10
        )
        assert len(result) == 4
    
    def test_simulate_match_payoffs_positive(self, engine):
        """Test match payoffs are non-negative."""
        result = engine._simulate_match(
            StrategyType.TIT_FOR_TAT, 
            StrategyType.ALWAYS_DEFECT, 
            10
        )
        assert result[0] >= 0
        assert result[1] >= 0
    
    def test_simulate_match_coop_rates_valid(self, engine):
        """Test cooperation rates are between 0 and 1."""
        result = engine._simulate_match(
            StrategyType.RANDOM, 
            StrategyType.RANDOM, 
            100
        )
        assert 0 <= result[2] <= 1
        assert 0 <= result[3] <= 1
    
    def test_tournament_returns_dataframe(self, engine):
        """Test tournament returns DataFrame."""
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE]
        results = engine.run_tournament(strategies, rounds_per_match=10)
        assert isinstance(results, pd.DataFrame)
    
    def test_tournament_correct_number_of_matchups(self, engine):
        """Test tournament has n^2 matchups."""
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE, 
                     StrategyType.ALWAYS_DEFECT]
        results = engine.run_tournament(strategies, rounds_per_match=10)
        assert len(results) == len(strategies) ** 2
    
    def test_tournament_has_required_columns(self, engine):
        """Test tournament results have all required columns."""
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE]
        results = engine.run_tournament(strategies, rounds_per_match=10)
        
        required_cols = ['Strategy_1', 'Strategy_2', 'Payoff_1', 'Payoff_2',
                        'Avg_Payoff_Per_Round_1', 'Avg_Payoff_Per_Round_2',
                        'Cooperation_Rate_1', 'Cooperation_Rate_2', 'Rounds']
        for col in required_cols:
            assert col in results.columns, f"Missing column: {col}"
    
    def test_tournament_per_round_payoff_calculation(self, engine):
        """Test per-round payoff is correctly calculated (BUG-016)."""
        strategies = [StrategyType.TIT_FOR_TAT]
        rounds = 10
        results = engine.run_tournament(strategies, rounds_per_match=rounds)
        
        for _, row in results.iterrows():
            expected = row['Payoff_1'] / rounds
            assert abs(row['Avg_Payoff_Per_Round_1'] - expected) < 0.001
    
    def test_evolutionary_simulation_returns_dataframe(self, engine):
        """Test evolutionary simulation returns DataFrame."""
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 5,
            StrategyType.ALWAYS_DEFECT: 5
        }
        results = engine.run_evolutionary_simulation(initial_pop, generations=5)
        assert isinstance(results, pd.DataFrame)
    
    def test_evolutionary_simulation_correct_generations(self, engine):
        """Test evolutionary simulation runs for correct number of generations."""
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 5,
            StrategyType.ALWAYS_DEFECT: 5
        }
        generations = 10
        results = engine.run_evolutionary_simulation(initial_pop, generations=generations)
        assert len(results) == generations
    
    def test_stochastic_game_returns_dataframe(self, engine):
        """Test stochastic game returns DataFrame."""
        results = engine.run_stochastic_game(rounds=20)
        assert isinstance(results, pd.DataFrame)
    
    def test_stochastic_game_valid_states(self, engine):
        """Test stochastic game states are valid."""
        results = engine.run_stochastic_game(rounds=50)
        valid_states = ['Cooperative', 'Neutral', 'Hostile']
        for state in results['State']:
            assert state in valid_states
    
    def test_stochastic_game_probability_assertions(self, engine):
        """Test stochastic game state transitions don't raise assertion errors (BUG-011)."""
        # Run many rounds to test probability handling
        results = engine.run_stochastic_game(rounds=200)
        assert len(results) == 200
    
    def test_learning_simulation_fictitious_play(self, engine):
        """Test fictitious play learning."""
        results = engine.run_learning_simulation('fictitious_play', rounds=20)
        assert 'US_Belief_China_Coop' in results.columns
        assert 'China_Belief_US_Coop' in results.columns
    
    def test_learning_simulation_reinforcement(self, engine):
        """Test reinforcement learning."""
        results = engine.run_learning_simulation('reinforcement', rounds=20)
        assert 'US_Q_Coop' in results.columns
        assert 'US_Q_Defect' in results.columns
    
    def test_learning_simulation_regret_matching(self, engine):
        """Test regret matching."""
        results = engine.run_learning_simulation('regret_matching', rounds=50)
        assert 'US_Regret_Coop' in results.columns
    
    def test_regret_matching_bounded(self, engine):
        """Test regret values are bounded (BUG-010)."""
        results = engine.run_learning_simulation('regret_matching', rounds=200)
        max_regret = 10000.0
        assert results['US_Regret_Coop'].max() <= max_regret
        assert results['US_Regret_Defect'].max() <= max_regret


# =============================================================================
# UNIT TESTS - Strategy Actions
# =============================================================================

class TestStrategyActions:
    """Test individual strategy implementations."""
    
    def test_always_cooperate_first_round(self, engine):
        """Test Always Cooperate on first round."""
        action = engine._get_strategy_action(StrategyType.ALWAYS_COOPERATE, 0, [], [], [])
        assert action == 'C'
    
    def test_always_cooperate_after_defection(self, engine):
        """Test Always Cooperate after opponent defects."""
        action = engine._get_strategy_action(StrategyType.ALWAYS_COOPERATE, 5, 
                                            ['C']*5, ['D']*5, [0]*5)
        assert action == 'C'
    
    def test_always_defect_first_round(self, engine):
        """Test Always Defect on first round."""
        action = engine._get_strategy_action(StrategyType.ALWAYS_DEFECT, 0, [], [], [])
        assert action == 'D'
    
    def test_always_defect_after_cooperation(self, engine):
        """Test Always Defect after opponent cooperates."""
        action = engine._get_strategy_action(StrategyType.ALWAYS_DEFECT, 5,
                                            ['D']*5, ['C']*5, [5]*5)
        assert action == 'D'
    
    def test_tft_first_round(self, engine):
        """Test Tit-for-Tat cooperates first."""
        action = engine._get_strategy_action(StrategyType.TIT_FOR_TAT, 0, [], [], [])
        assert action == 'C'
    
    def test_tft_mirrors_cooperation(self, engine):
        """Test TFT mirrors cooperation."""
        action = engine._get_strategy_action(StrategyType.TIT_FOR_TAT, 1, ['C'], ['C'], [3])
        assert action == 'C'
    
    def test_tft_mirrors_defection(self, engine):
        """Test TFT mirrors defection."""
        action = engine._get_strategy_action(StrategyType.TIT_FOR_TAT, 1, ['C'], ['D'], [0])
        assert action == 'D'
    
    def test_grim_trigger_cooperates_initially(self, engine):
        """Test Grim Trigger cooperates initially."""
        action = engine._get_strategy_action(StrategyType.GRIM_TRIGGER, 0, [], [], [])
        assert action == 'C'
    
    def test_grim_trigger_after_cooperation(self, engine):
        """Test Grim Trigger continues cooperating."""
        action = engine._get_strategy_action(StrategyType.GRIM_TRIGGER, 3,
                                            ['C']*3, ['C']*3, [3]*3)
        assert action == 'C'
    
    def test_grim_trigger_after_defection(self, engine):
        """Test Grim Trigger defects forever after opponent defects."""
        action = engine._get_strategy_action(StrategyType.GRIM_TRIGGER, 3,
                                            ['C', 'C', 'D'], ['C', 'D', 'D'], [3, 0, 1])
        assert action == 'D'
    
    def test_pavlov_first_round(self, engine):
        """Test Pavlov cooperates first."""
        action = engine._get_strategy_action(StrategyType.PAVLOV, 0, [], [], [])
        assert action == 'C'
    
    def test_pavlov_win_stay(self, engine):
        """Test Pavlov stays after winning (high payoff)."""
        action = engine._get_strategy_action(StrategyType.PAVLOV, 1, ['C'], ['C'], [3])
        assert action == 'C'
    
    def test_pavlov_lose_shift(self, engine):
        """Test Pavlov shifts after losing (low payoff)."""
        action = engine._get_strategy_action(StrategyType.PAVLOV, 1, ['C'], ['D'], [0])
        assert action == 'D'
    
    def test_random_returns_valid_action(self, engine):
        """Test Random returns C or D."""
        actions = set()
        for _ in range(100):
            action = engine._get_strategy_action(StrategyType.RANDOM, 0, [], [], [])
            actions.add(action)
        assert actions == {'C', 'D'}


# =============================================================================
# UNIT TESTS - GameTheoryEngine
# =============================================================================

class TestGameTheoryEngine:
    """Unit tests for game theory engine."""
    
    def test_find_nash_equilibria(self, game_engine):
        """Test Nash equilibrium detection."""
        equilibria = game_engine.find_nash_equilibria()
        assert isinstance(equilibria, list)
        assert len(equilibria) > 0
        # PD has (Defect, Defect) as NE
        assert 'defect' in str(equilibria).lower()
    
    def test_find_dominant_strategies(self, game_engine):
        """Test dominant strategy detection."""
        dominant = game_engine.find_dominant_strategies()
        assert isinstance(dominant, dict)
    
    def test_analyze_game(self, game_engine):
        """Test game engine has analysis methods."""
        # Verify engine has key methods
        assert hasattr(game_engine, 'find_nash_equilibria')
        assert hasattr(game_engine, 'find_dominant_strategies')
        assert hasattr(game_engine, 'matrix')


# =============================================================================
# UNIT TESTS - Visualization
# =============================================================================

class TestVisualization:
    """Test visualization components."""
    
    def test_tournament_heatmap_creation(self, engine):
        """Test tournament heatmap figure creation."""
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE,
                     StrategyType.ALWAYS_DEFECT]
        results = engine.run_tournament(strategies, rounds_per_match=10)
        
        fig = AdvancedVisualizationEngine.create_tournament_heatmap(results)
        assert fig is not None
        assert hasattr(fig, 'data')
    
    def test_tournament_rankings_creation(self, engine):
        """Test tournament rankings figure creation."""
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE,
                     StrategyType.ALWAYS_DEFECT]
        results = engine.run_tournament(strategies, rounds_per_match=10)
        
        fig = AdvancedVisualizationEngine.create_tournament_rankings(results)
        assert fig is not None
    
    def test_evolutionary_dynamics_chart(self, engine):
        """Test evolutionary dynamics chart creation."""
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 10,
            StrategyType.ALWAYS_DEFECT: 10
        }
        results = engine.run_evolutionary_simulation(initial_pop, generations=10)
        
        fig = AdvancedVisualizationEngine.create_evolutionary_dynamics_chart(results)
        assert fig is not None
    
    def test_learning_dynamics_chart(self, engine):
        """Test learning dynamics chart creation."""
        results = engine.run_learning_simulation('fictitious_play', rounds=20)
        
        fig = AdvancedVisualizationEngine.create_learning_dynamics_chart(
            results, 'Fictitious Play'
        )
        assert fig is not None


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for component interactions."""
    
    def test_full_tournament_pipeline(self, pd_matrix):
        """Test complete tournament workflow."""
        config = AdvancedSimulationConfig(rounds=50, noise_probability=0.01)
        engine = AdvancedSimulationEngine(pd_matrix, config)
        
        strategies = [StrategyType.TIT_FOR_TAT, StrategyType.ALWAYS_COOPERATE,
                     StrategyType.ALWAYS_DEFECT, StrategyType.GRIM_TRIGGER]
        results = engine.run_tournament(strategies, rounds_per_match=50)
        
        # Verify results
        assert len(results) == 16
        assert results['Payoff_1'].sum() > 0
        
        # Verify rankings can be computed
        rankings = results.groupby('Strategy_1')['Payoff_1'].sum().sort_values(ascending=False)
        assert len(rankings) == 4
        
        # Verify visualization works
        fig = AdvancedVisualizationEngine.create_tournament_rankings(results)
        assert fig is not None
    
    def test_evolutionary_to_equilibrium(self, pd_matrix):
        """Test evolutionary dynamics maintains population."""
        config = AdvancedSimulationConfig(generations=30, population_size=100, 
                                         mutation_rate=0.01)
        engine = AdvancedSimulationEngine(pd_matrix, config)
        
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 25,
            StrategyType.ALWAYS_COOPERATE: 25,
            StrategyType.ALWAYS_DEFECT: 25,
            StrategyType.GRIM_TRIGGER: 25
        }
        
        results = engine.run_evolutionary_simulation(initial_pop, generations=30)
        
        # Verify population is maintained
        last_gen = results.iloc[-1]
        total_pop = sum([last_gen.get(s.value, 0) for s in initial_pop.keys()])
        assert total_pop >= 50
    
    def test_learning_convergence(self, pd_matrix):
        """Test learning algorithms show convergence."""
        config = AdvancedSimulationConfig(rounds=200, learning_rate=0.1)
        engine = AdvancedSimulationEngine(pd_matrix, config)
        
        results = engine.run_learning_simulation('fictitious_play', rounds=200)
        
        # Check beliefs stabilize (late variance <= early variance)
        early_var = results.iloc[:20]['US_Belief_China_Coop'].var()
        late_var = results.iloc[-20:]['US_Belief_China_Coop'].var()
        assert late_var <= early_var + 0.15
    
    def test_matrix_to_engine_to_viz(self, pd_matrix, basic_config):
        """Test data flows correctly through pipeline."""
        engine = AdvancedSimulationEngine(pd_matrix, basic_config)
        
        # Simulation
        results = engine.run_stochastic_game(rounds=50)
        
        # Verify data integrity
        assert len(results) == 50
        assert 'US_Payoff' in results.columns
        assert results['US_Payoff'].notna().all()


# =============================================================================
# FUNCTIONAL TESTS - Features
# =============================================================================

class TestFeatures:
    """Functional tests for app features."""
    
    def test_discount_factor_calculation(self, pd_matrix):
        """Test critical discount factor formula."""
        params = pd_matrix.get_payoff_parameters_row()
        T, R, P = params['T'], params['R'], params['P']
        
        # Folk theorem discount factor
        delta_star = (T - R) / (T - P)
        
        assert 0 < delta_star < 1
        assert delta_star == 0.5  # For standard PD
    
    def test_game_classification_pd(self, pd_matrix):
        """Test Prisoner's Dilemma is correctly classified."""
        params = pd_matrix.get_payoff_parameters_row()
        
        # PD: T > R > P > S
        assert params['T'] > params['R']
        assert params['R'] > params['P']
        assert params['P'] > params['S']
    
    def test_game_classification_harmony(self, harmony_matrix):
        """Test Harmony Game structure."""
        params = harmony_matrix.get_payoff_parameters_row()
        
        # Harmony: R > T
        assert params['R'] > params['T']
    
    def test_all_strategies_executable(self, engine):
        """Test all strategy types can be executed."""
        all_strategies = [
            StrategyType.ALWAYS_COOPERATE,
            StrategyType.ALWAYS_DEFECT,
            StrategyType.TIT_FOR_TAT,
            StrategyType.GRIM_TRIGGER,
            StrategyType.PAVLOV,
            StrategyType.RANDOM,
        ]
        
        for strat in all_strategies:
            action = engine._get_strategy_action(strat, 0, [], [], [])
            assert action in ['C', 'D'], f"Strategy {strat.value} returned invalid action"
    
    def test_noise_affects_outcomes(self, pd_matrix):
        """Test that noise probability affects match outcomes."""
        np.random.seed(42)
        
        # Without noise
        config1 = AdvancedSimulationConfig(rounds=100, noise_probability=0.0)
        engine1 = AdvancedSimulationEngine(pd_matrix, config1)
        result1 = engine1._simulate_match(StrategyType.TIT_FOR_TAT, 
                                          StrategyType.TIT_FOR_TAT, 100)
        
        np.random.seed(42)
        
        # With high noise
        config2 = AdvancedSimulationConfig(rounds=100, noise_probability=0.3)
        engine2 = AdvancedSimulationEngine(pd_matrix, config2)
        result2 = engine2._simulate_match(StrategyType.TIT_FOR_TAT,
                                          StrategyType.TIT_FOR_TAT, 100)
        
        # Noise should affect cooperation rates
        assert result1[2] != result2[2] or result1[3] != result2[3]


# =============================================================================
# USER ACCEPTANCE TESTS
# =============================================================================

class TestUserAcceptance:
    """User acceptance tests simulating real-world scenarios."""
    
    def test_scenario_trade_war_analysis(self):
        """UAT: Model trade war as Prisoner's Dilemma."""
        trade_matrix = PayoffMatrix(
            cc=(8, 8), cd=(2, 10), dc=(10, 2), dd=(4, 4)
        )
        
        engine = GameTheoryEngine(trade_matrix)
        params = trade_matrix.get_payoff_parameters_row()
        
        # Verify PD structure
        assert params['T'] > params['R'] > params['P'] > params['S']
        
        # Find equilibrium
        equilibria = engine.find_nash_equilibria()
        assert len(equilibria) > 0
    
    def test_scenario_cooperation_sustainability(self, pd_matrix):
        """UAT: Test TFT sustains cooperation."""
        config = AdvancedSimulationConfig(rounds=100)
        engine = AdvancedSimulationEngine(pd_matrix, config)
        
        # TFT vs TFT should achieve high cooperation
        result = engine._simulate_match(StrategyType.TIT_FOR_TAT,
                                        StrategyType.TIT_FOR_TAT, 100)
        
        # Cooperation rate should be high
        assert result[2] > 0.9
        assert result[3] > 0.9
    
    def test_scenario_defection_dominance(self, pd_matrix):
        """UAT: Always Defect dominates Always Cooperate."""
        config = AdvancedSimulationConfig(rounds=100)
        engine = AdvancedSimulationEngine(pd_matrix, config)
        
        result = engine._simulate_match(StrategyType.ALWAYS_DEFECT,
                                        StrategyType.ALWAYS_COOPERATE, 100)
        
        # Defector should have higher payoff
        assert result[0] > result[1]
    
    def test_scenario_evolutionary_selection(self, pd_matrix):
        """UAT: Test evolutionary selection pressure."""
        config = AdvancedSimulationConfig(generations=50, population_size=100)
        engine = AdvancedSimulationEngine(pd_matrix, config)
        
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 50,
            StrategyType.ALWAYS_DEFECT: 50
        }
        
        results = engine.run_evolutionary_simulation(initial_pop, generations=50)
        
        # Should have data for all generations
        assert len(results) == 50
    
    def test_scenario_visualization_data(self, pd_matrix):
        """UAT: Generate valid visualization data."""
        config = AdvancedSimulationConfig(generations=50, population_size=100)
        engine = AdvancedSimulationEngine(pd_matrix, config)
        
        initial_pop = {
            StrategyType.TIT_FOR_TAT: 25,
            StrategyType.ALWAYS_COOPERATE: 25,
            StrategyType.ALWAYS_DEFECT: 25,
            StrategyType.GRIM_TRIGGER: 25
        }
        
        results = engine.run_evolutionary_simulation(initial_pop, generations=50)
        
        # Verify share columns for plotting
        share_cols = [col for col in results.columns if '_Share' in col]
        assert len(share_cols) >= 4
        
        # Verify shares approximately sum to 1
        for _, row in results.iterrows():
            total = sum([row[col] for col in share_cols])
            assert 0.8 <= total <= 1.2, f"Shares sum to {total}"
    
    def test_scenario_comparative_statics(self):
        """UAT: Test comparative statics on payoff parameters."""
        # Base case
        matrix1 = PayoffMatrix(cc=(3, 3), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        params1 = matrix1.get_payoff_parameters_row()
        delta1 = (params1['T'] - params1['R']) / (params1['T'] - params1['P'])
        
        # Increase R (reward)
        matrix2 = PayoffMatrix(cc=(4, 4), cd=(0, 5), dc=(5, 0), dd=(1, 1))
        params2 = matrix2.get_payoff_parameters_row()
        delta2 = (params2['T'] - params2['R']) / (params2['T'] - params2['P'])
        
        # Higher R should lower discount factor
        assert delta2 < delta1


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_rounds_tournament(self, engine):
        """Test tournament with minimal rounds."""
        strategies = [StrategyType.TIT_FOR_TAT]
        results = engine.run_tournament(strategies, rounds_per_match=1)
        assert len(results) == 1
    
    def test_single_strategy_tournament(self, engine):
        """Test tournament with single strategy (self-play)."""
        strategies = [StrategyType.ALWAYS_COOPERATE]
        results = engine.run_tournament(strategies, rounds_per_match=10)
        assert len(results) == 1
    
    def test_empty_history_strategy_action(self, engine):
        """Test strategy action with empty history."""
        for strat in [StrategyType.TIT_FOR_TAT, StrategyType.GRIM_TRIGGER,
                     StrategyType.PAVLOV]:
            action = engine._get_strategy_action(strat, 0, [], [], [])
            assert action in ['C', 'D']
    
    def test_long_history_strategy_action(self, engine):
        """Test strategy action with long history."""
        history = ['C', 'D'] * 500
        payoffs = [3, 0] * 500
        
        action = engine._get_strategy_action(StrategyType.TIT_FOR_TAT, 
                                            1000, history, history, payoffs)
        assert action in ['C', 'D']
    
    def test_extreme_payoffs(self):
        """Test matrix with extreme payoff values."""
        matrix = PayoffMatrix(cc=(1000, 1000), cd=(0, 10000), 
                             dc=(10000, 0), dd=(1, 1))
        params = matrix.get_payoff_parameters_row()
        assert params['T'] == 10000
        assert params['R'] == 1000
    
    def test_negative_payoffs(self):
        """Test matrix with negative payoffs."""
        matrix = PayoffMatrix(cc=(0, 0), cd=(-5, 5), dc=(5, -5), dd=(-1, -1))
        params = matrix.get_payoff_parameters_row()
        assert params['S'] == -5
        assert params['P'] == -1


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-x'])
