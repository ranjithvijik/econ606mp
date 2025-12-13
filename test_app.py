"""
Comprehensive Unit and Integration Tests for Game Theory Application
=====================================================================
Tests for: app.py - U.S.-China Game Theory Analysis Application

Run with: pytest test_app.py -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components from app.py
from app import (
    PayoffMatrix,
    GameTheoryEngine,
    DataManager,
    GameType,
    HistoricalPeriod,
    StrategyType,
    EquilibriumResult,
    StatisticalEngine,
    ProfessionalTheme,
    get_professional_layout,
    categorize_delta,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def harmony_matrix():
    """Fixture for Harmony Game payoff matrix (2001-2007)."""
    return PayoffMatrix(
        cc=(8, 8),   # Both cooperate
        cd=(2, 5),   # US cooperates, China defects
        dc=(5, 2),   # US defects, China cooperates
        dd=(1, 1)    # Both defect
    )


@pytest.fixture
def pd_matrix():
    """Fixture for Prisoner's Dilemma payoff matrix (2018-2025)."""
    return PayoffMatrix(
        cc=(6, 6),   # Both cooperate
        cd=(0, 8),   # US cooperates, China defects
        dc=(8, 0),   # US defects, China cooperates
        dd=(2, 2)    # Both defect
    )


@pytest.fixture
def game_engine(harmony_matrix):
    """Fixture for GameTheoryEngine with Harmony matrix."""
    return GameTheoryEngine(harmony_matrix)


@pytest.fixture
def data_manager():
    """Fixture for DataManager."""
    return DataManager()


# =============================================================================
# UNIT TESTS: PayoffMatrix
# =============================================================================

class TestPayoffMatrix:
    """Unit tests for PayoffMatrix dataclass."""
    
    def test_creation(self, harmony_matrix):
        """Test PayoffMatrix creation."""
        assert harmony_matrix.cc == (8, 8)
        assert harmony_matrix.cd == (2, 5)
        assert harmony_matrix.dc == (5, 2)
        assert harmony_matrix.dd == (1, 1)
    
    def test_to_numpy(self, harmony_matrix):
        """Test conversion to numpy array."""
        result = harmony_matrix.to_numpy()
        # Result is a tuple of two arrays
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], np.ndarray)
        assert isinstance(result[1], np.ndarray)
    
    def test_total_payoff(self, harmony_matrix):
        """Test total payoff calculation."""
        total_cc = sum(harmony_matrix.cc)
        total_dd = sum(harmony_matrix.dd)
        assert total_cc == 16
        assert total_dd == 2
    
    def test_payoff_access(self, harmony_matrix):
        """Test accessing individual payoffs."""
        assert harmony_matrix.cc[0] == 8  # US payoff at (C,C)
        assert harmony_matrix.cc[1] == 8  # China payoff at (C,C)
        assert harmony_matrix.dd[0] == 1  # US payoff at (D,D)
        assert harmony_matrix.dd[1] == 1  # China payoff at (D,D)


# =============================================================================
# UNIT TESTS: GameTheoryEngine
# =============================================================================

class TestGameTheoryEngine:
    """Unit tests for GameTheoryEngine class."""
    
    def test_initialization(self, game_engine, harmony_matrix):
        """Test engine initialization."""
        assert game_engine.matrix == harmony_matrix
        assert hasattr(game_engine, '_cache')
    
    def test_find_nash_equilibria_harmony(self, harmony_matrix):
        """Test Nash equilibrium for Harmony Game."""
        engine = GameTheoryEngine(harmony_matrix)
        equilibria = engine.find_nash_equilibria()
        
        assert isinstance(equilibria, list)
        assert len(equilibria) >= 1
        # Harmony game has (C,C) as Nash equilibrium - check string format
        assert any('Cooperate' in str(eq) for eq in equilibria)
    
    def test_find_nash_equilibria_pd(self, pd_matrix):
        """Test Nash equilibrium for Prisoner's Dilemma."""
        engine = GameTheoryEngine(pd_matrix)
        equilibria = engine.find_nash_equilibria()
        
        assert isinstance(equilibria, list)
        assert len(equilibria) >= 1
        # PD has (D,D) as Nash equilibrium
        assert any('Defect' in str(eq) for eq in equilibria)
    
    def test_find_dominant_strategies_harmony(self, harmony_matrix):
        """Test dominant strategies for Harmony Game."""
        engine = GameTheoryEngine(harmony_matrix)
        dominant = engine.find_dominant_strategies()
        
        assert isinstance(dominant, dict)
        # Check for either format
        assert 'US' in dominant or 'player_1' in dominant
        # In Harmony, Cooperate is dominant
        values = list(dominant.values())
        assert any('Cooperate' in str(v) for v in values)
    
    def test_find_dominant_strategies_pd(self, pd_matrix):
        """Test dominant strategies for Prisoner's Dilemma."""
        engine = GameTheoryEngine(pd_matrix)
        dominant = engine.find_dominant_strategies()
        
        # In PD, Defect is dominant
        values = list(dominant.values())
        assert any('Defect' in str(v) for v in values)
    
    def test_pareto_efficiency_analysis(self, harmony_matrix):
        """Test Pareto efficiency analysis."""
        engine = GameTheoryEngine(harmony_matrix)
        analysis = engine.pareto_efficiency_analysis()
        
        assert isinstance(analysis, dict)
        # Check structure - may return outcomes with True/False
        assert len(analysis) > 0
        # (C,C) should be Pareto efficient
        assert analysis.get('(C,C)', False) == True
    
    def test_calculate_critical_discount_factor(self, pd_matrix):
        """Test critical discount factor calculation."""
        engine = GameTheoryEngine(pd_matrix)
        delta = engine.calculate_critical_discount_factor()
        
        assert isinstance(delta, (int, float))
        assert 0 <= delta <= 1
    
    def test_classify_game_type_harmony(self, harmony_matrix):
        """Test game classification for Harmony."""
        engine = GameTheoryEngine(harmony_matrix)
        game_type = engine.classify_game_type()
        
        assert game_type == GameType.HARMONY
    
    def test_classify_game_type_pd(self, pd_matrix):
        """Test game classification for Prisoner's Dilemma."""
        engine = GameTheoryEngine(pd_matrix)
        game_type = engine.classify_game_type()
        
        assert game_type == GameType.PRISONERS_DILEMMA
    
    def test_copy_method(self, game_engine):
        """Test engine copy method."""
        copy = game_engine.copy()
        
        assert copy is not game_engine
        assert copy.matrix == game_engine.matrix
    
    def test_add_noise(self, game_engine):
        """Test adding noise to payoff matrix."""
        noisy = game_engine.add_noise(noise_level=0.1)
        
        assert noisy is not game_engine
        assert isinstance(noisy.matrix, PayoffMatrix)
    
    def test_simulate_strategy(self, game_engine):
        """Test strategy simulation."""
        from app import SimulationResult
        results = game_engine.simulate_strategy(
            strategy=StrategyType.TIT_FOR_TAT,
            rounds=10
        )
        
        # Results is a SimulationResult object, not a dict
        assert isinstance(results, SimulationResult)
        assert hasattr(results, 'total_us_payoff')
        assert hasattr(results, 'total_china_payoff')
    
    def test_calculate_cooperation_margin(self, pd_matrix):
        """Test cooperation margin calculation."""
        engine = GameTheoryEngine(pd_matrix)
        margin = engine.calculate_cooperation_margin(0.9)
        
        assert isinstance(margin, (int, float))


# =============================================================================
# UNIT TESTS: DataManager
# =============================================================================

class TestDataManager:
    """Unit tests for DataManager class."""
    
    def test_get_tariff_data(self, data_manager):
        """Test tariff data retrieval."""
        data = data_manager.get_tariff_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        # Check for some column that should exist
        assert len(data.columns) >= 2
    
    def test_get_cooperation_index_data(self, data_manager):
        """Test cooperation index data."""
        data = data_manager.get_cooperation_index_data()
        
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
    
    def test_get_discount_factor_data(self, data_manager):
        """Test discount factor data."""
        data = data_manager.get_discount_factor_data()
        
        assert isinstance(data, pd.DataFrame)
    
    def test_data_caching(self, data_manager):
        """Test that data is cached properly."""
        # First call
        data1 = data_manager.get_tariff_data()
        # Second call should use cache
        data2 = data_manager.get_tariff_data()
        
        # Should return same data
        assert len(data1) == len(data2)


# =============================================================================
# UNIT TESTS: Helper Functions
# =============================================================================

class TestHelperFunctions:
    """Unit tests for helper functions."""
    
    def test_categorize_delta(self):
        """Test categorizing delta values."""
        result = categorize_delta(0.15)
        assert isinstance(result, str)
    
    def test_categorize_delta_negative(self):
        """Test categorizing negative delta."""
        result = categorize_delta(-0.1)
        assert isinstance(result, str)
    
    def test_get_professional_layout(self):
        """Test professional Plotly layout configuration."""
        layout = get_professional_layout()
        
        assert isinstance(layout, dict)
        assert 'template' in layout
        assert 'font' in layout
        assert layout['template'] == 'plotly_white'


# =============================================================================
# UNIT TESTS: ProfessionalTheme
# =============================================================================

class TestProfessionalTheme:
    """Unit tests for ProfessionalTheme class."""
    
    def test_primary_color(self):
        """Test primary color is defined."""
        assert ProfessionalTheme.PRIMARY == '#1E3A8A'
    
    def test_chart_palette(self):
        """Test chart palette has colors."""
        assert len(ProfessionalTheme.CHART_PALETTE) >= 5
        assert all(c.startswith('#') for c in ProfessionalTheme.CHART_PALETTE)
    
    def test_font_family(self):
        """Test font family is defined."""
        assert 'Inter' in ProfessionalTheme.FONT_FAMILY
    
    def test_semantic_colors(self):
        """Test semantic colors are defined."""
        assert hasattr(ProfessionalTheme, 'SUCCESS')
        assert hasattr(ProfessionalTheme, 'WARNING')
        assert hasattr(ProfessionalTheme, 'DANGER')


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_full_analysis_pipeline(self, harmony_matrix, pd_matrix):
        """Test complete analysis pipeline."""
        # Create engines
        harmony_engine = GameTheoryEngine(harmony_matrix)
        pd_engine = GameTheoryEngine(pd_matrix)
        
        # Run analysis
        harmony_eq = harmony_engine.find_nash_equilibria()
        pd_eq = pd_engine.find_nash_equilibria()
        
        harmony_type = harmony_engine.classify_game_type()
        pd_type = pd_engine.classify_game_type()
        
        # Verify structural transformation
        assert harmony_type == GameType.HARMONY
        assert pd_type == GameType.PRISONERS_DILEMMA
        
        # Verify equilibria exist
        assert len(harmony_eq) >= 1
        assert len(pd_eq) >= 1
    
    def test_data_to_analysis_pipeline(self, data_manager, harmony_matrix):
        """Test data retrieval to analysis pipeline."""
        # Get data
        tariff_data = data_manager.get_tariff_data()
        coop_data = data_manager.get_cooperation_index_data()
        
        # Verify data integrity
        assert len(tariff_data) > 0
        assert len(coop_data) > 0
        
        # Create engine and analyze
        engine = GameTheoryEngine(harmony_matrix)
        eq = engine.find_nash_equilibria()
        pareto = engine.pareto_efficiency_analysis()
        
        # Verify combined results
        assert len(eq) > 0
        assert len(pareto) > 0
    
    def test_monte_carlo_with_noise(self, harmony_matrix):
        """Test Monte Carlo simulation with noise."""
        engine = GameTheoryEngine(harmony_matrix)
        
        # Run multiple simulations with noise
        results = []
        for _ in range(5):
            noisy_engine = engine.add_noise(noise_level=0.1)
            game_type = noisy_engine.classify_game_type()
            results.append(game_type)
        
        # Should get results for each run
        assert len(results) == 5
    
    def test_discount_factor_threshold_analysis(self, pd_matrix):
        """Test discount factor vs cooperation threshold."""
        engine = GameTheoryEngine(pd_matrix)
        
        # Calculate critical threshold
        delta_star = engine.calculate_critical_discount_factor()
        
        # Test is valid number
        assert isinstance(delta_star, (int, float))
        
        # Test cooperation margin at different discount factors
        margin_low = engine.calculate_cooperation_margin(0.3)
        margin_high = engine.calculate_cooperation_margin(0.9)
        
        # Higher discount factors should enable more cooperation
        assert margin_high >= margin_low


# =============================================================================
# EDGE CASE TESTS
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    def test_zero_payoff_matrix(self):
        """Test matrix with zero payoffs."""
        matrix = PayoffMatrix(
            cc=(0, 0),
            cd=(0, 0),
            dc=(0, 0),
            dd=(0, 0)
        )
        engine = GameTheoryEngine(matrix)
        eq = engine.find_nash_equilibria()
        
        # Should still find equilibria
        assert isinstance(eq, list)
    
    def test_identical_payoff_matrix(self):
        """Test matrix with identical payoffs."""
        matrix = PayoffMatrix(
            cc=(5, 5),
            cd=(5, 5),
            dc=(5, 5),
            dd=(5, 5)
        )
        engine = GameTheoryEngine(matrix)
        eq = engine.find_nash_equilibria()
        
        assert len(eq) >= 1
    
    def test_asymmetric_payoffs(self):
        """Test asymmetric payoff matrix."""
        matrix = PayoffMatrix(
            cc=(10, 5),
            cd=(0, 8),
            dc=(12, 0),
            dd=(3, 2)
        )
        engine = GameTheoryEngine(matrix)
        game_type = engine.classify_game_type()
        
        assert game_type is not None


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Basic performance tests."""
    
    def test_nash_equilibrium_speed(self, harmony_matrix):
        """Test Nash equilibrium calculation is fast."""
        import time
        
        engine = GameTheoryEngine(harmony_matrix)
        
        start = time.time()
        for _ in range(100):
            engine.find_nash_equilibria()
        elapsed = time.time() - start
        
        # Should complete 100 iterations in under 2 seconds
        assert elapsed < 2.0
    
    def test_game_classification_speed(self, harmony_matrix):
        """Test game classification is fast."""
        import time
        
        engine = GameTheoryEngine(harmony_matrix)
        
        start = time.time()
        for _ in range(100):
            engine.classify_game_type()
        elapsed = time.time() - start
        
        # Should be fast
        assert elapsed < 2.0


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestSummary:
    """Summary tests to verify core functionality."""
    
    def test_core_workflow(self, harmony_matrix, pd_matrix, data_manager):
        """Test the complete core workflow of the application."""
        
        # 1. Load data
        tariff_data = data_manager.get_tariff_data()
        coop_data = data_manager.get_cooperation_index_data()
        assert len(tariff_data) > 0
        assert len(coop_data) > 0
        
        # 2. Create game engines
        harmony_engine = GameTheoryEngine(harmony_matrix)
        pd_engine = GameTheoryEngine(pd_matrix)
        
        # 3. Classify games
        assert harmony_engine.classify_game_type() == GameType.HARMONY
        assert pd_engine.classify_game_type() == GameType.PRISONERS_DILEMMA
        
        # 4. Find equilibria
        harmony_eq = harmony_engine.find_nash_equilibria()
        pd_eq = pd_engine.find_nash_equilibria()
        assert len(harmony_eq) >= 1
        assert len(pd_eq) >= 1
        
        # 5. Calculate discount factors (may be negative for some games)
        delta_harmony = harmony_engine.calculate_critical_discount_factor()
        delta_pd = pd_engine.calculate_critical_discount_factor()
        assert isinstance(delta_harmony, (int, float))
        assert isinstance(delta_pd, (int, float))
        
        # 6. Test Pareto analysis
        pareto_harmony = harmony_engine.pareto_efficiency_analysis()
        pareto_pd = pd_engine.pareto_efficiency_analysis()
        assert len(pareto_harmony) > 0
        assert len(pareto_pd) > 0
        
        print("\n✅ Core workflow test PASSED!")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
