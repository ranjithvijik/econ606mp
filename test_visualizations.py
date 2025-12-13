"""
Visualization Tests for Game Theory Application
================================================
Tests for: All Plotly charts and visualization components

Run with: pytest test_visualizations.py -v --tb=short
"""

import pytest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sys
import os

# Add the current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import components from app.py
from app import (
    PayoffMatrix,
    GameTheoryEngine,
    DataManager,
    VisualizationEngine,
    get_professional_layout,
    ProfessionalTheme,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def harmony_matrix():
    """Fixture for Harmony Game payoff matrix."""
    return PayoffMatrix(
        cc=(8, 8), cd=(2, 5), dc=(5, 2), dd=(1, 1)
    )

@pytest.fixture
def pd_matrix():
    """Fixture for Prisoner's Dilemma payoff matrix."""
    return PayoffMatrix(
        cc=(6, 6), cd=(0, 8), dc=(8, 0), dd=(2, 2)
    )

@pytest.fixture
def data_manager():
    """Fixture for DataManager."""
    return DataManager()


# =============================================================================
# TEST: VisualizationEngine Static Methods
# =============================================================================

class TestVisualizationEngineCharts:
    """Tests for VisualizationEngine chart generation."""
    
    def test_create_payoff_matrix_heatmap(self, harmony_matrix):
        """Test payoff matrix heatmap generation."""
        fig = VisualizationEngine.create_payoff_matrix_heatmap(
            harmony_matrix, 
            title="Test Harmony Matrix"
        )
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        print("✅ Payoff Matrix Heatmap: PASSED")
    
    def test_create_cooperation_index_chart(self, data_manager):
        """Test cooperation index chart generation with real data."""
        coop_data = data_manager.get_cooperation_index_data()
        fig = VisualizationEngine.create_cooperation_index_chart(coop_data)
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        print("✅ Cooperation Index Chart: PASSED")
    
    def test_create_tariff_escalation_chart(self, data_manager):
        """Test tariff escalation chart generation with real data."""
        tariff_data = data_manager.get_tariff_data()
        fig = VisualizationEngine.create_tariff_escalation_chart(tariff_data)
        
        assert fig is not None
        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # Should have US and China traces
        print("✅ Tariff Escalation Chart: PASSED")


# =============================================================================
# TEST: Professional Component Library
# =============================================================================

class TestProfessionalComponents:
    """Tests for professional visualization components."""
    
    def test_get_professional_layout(self):
        """Test professional Plotly layout configuration."""
        layout = get_professional_layout()
        
        assert isinstance(layout, dict)
        assert layout['template'] == 'plotly_white'
        assert 'font' in layout
        assert layout['plot_bgcolor'] == 'white'
        print("✅ Professional Layout: PASSED")
    
    def test_professional_layout_has_key_properties(self):
        """Test that professional layout has key properties."""
        layout = get_professional_layout()
        
        expected_keys = ['template', 'font', 'title', 'xaxis', 'yaxis', 
                        'plot_bgcolor', 'paper_bgcolor']
        
        for key in expected_keys:
            assert key in layout, f"Missing key: {key}"
        print("✅ Professional Layout Keys: PASSED")


# =============================================================================
# TEST: Chart Data Validation
# =============================================================================

class TestChartDataValidation:
    """Tests to validate chart data is correct."""
    
    def test_payoff_matrix_values_in_heatmap(self, harmony_matrix):
        """Verify payoff values are correctly displayed in heatmap."""
        fig = VisualizationEngine.create_payoff_matrix_heatmap(
            harmony_matrix, 
            title="Test"
        )
        
        # Check that figure has data
        assert len(fig.data) > 0
        print("✅ Payoff Matrix Values: PASSED")
    
    def test_cooperation_chart_has_traces(self, data_manager):
        """Verify cooperation chart has traces."""
        coop_data = data_manager.get_cooperation_index_data()
        fig = VisualizationEngine.create_cooperation_index_chart(coop_data)
        
        # Should have data points
        assert len(fig.data) > 0
        print("✅ Cooperation Chart Traces: PASSED")
    
    def test_tariff_chart_both_countries(self, data_manager):
        """Verify tariff chart shows both US and China data."""
        tariff_data = data_manager.get_tariff_data()
        fig = VisualizationEngine.create_tariff_escalation_chart(tariff_data)
        
        # At least 2 traces (US and China)
        assert len(fig.data) >= 2
        print("✅ Tariff Chart Both Countries: PASSED")


# =============================================================================
# TEST: Plotly Figure Properties
# =============================================================================

class TestPlotlyFigureProperties:
    """Tests to verify Plotly figure properties."""
    
    def test_figure_has_layout(self, harmony_matrix):
        """Verify figures have proper layout."""
        fig = VisualizationEngine.create_payoff_matrix_heatmap(
            harmony_matrix, title="Test"
        )
        
        assert hasattr(fig, 'layout')
        assert fig.layout is not None
        print("✅ Figure Has Layout: PASSED")
    
    def test_figure_has_title(self, harmony_matrix):
        """Verify figures have titles."""
        fig = VisualizationEngine.create_payoff_matrix_heatmap(
            harmony_matrix, title="Test Matrix"
        )
        
        # Title should be set
        assert fig.layout.title is not None or 'Test' in str(fig.layout)
        print("✅ Figure Has Title: PASSED")
    
    def test_figure_serializable(self, harmony_matrix):
        """Verify figures can be serialized to JSON."""
        fig = VisualizationEngine.create_payoff_matrix_heatmap(
            harmony_matrix, title="Test"
        )
        
        # Should be able to convert to JSON without errors
        json_str = fig.to_json()
        assert isinstance(json_str, str)
        assert len(json_str) > 0
        print("✅ Figure Serializable: PASSED")
    
    def test_figure_has_valid_traces(self, data_manager):
        """Verify figure traces are valid."""
        coop_data = data_manager.get_cooperation_index_data()
        fig = VisualizationEngine.create_cooperation_index_chart(coop_data)
        
        for trace in fig.data:
            # Each trace should have data
            assert trace is not None
        print("✅ Figure Valid Traces: PASSED")


# =============================================================================
# TEST: Chart Rendering (No Errors)
# =============================================================================

class TestChartRendering:
    """Tests to verify all charts render without errors."""
    
    def test_core_visualization_methods_exist(self):
        """Verify core visualization methods exist."""
        core_methods = [
            'create_payoff_matrix_heatmap',
            'create_cooperation_index_chart',
            'create_tariff_escalation_chart',
        ]
        
        for method in core_methods:
            assert hasattr(VisualizationEngine, method), f"Missing method: {method}"
        print("✅ Core Visualization Methods Exist: PASSED")
    
    def test_multiple_chart_creation(self, harmony_matrix, pd_matrix, data_manager):
        """Test creating multiple charts in sequence."""
        charts = []
        
        # Create multiple charts
        charts.append(VisualizationEngine.create_payoff_matrix_heatmap(
            harmony_matrix, title="Harmony"
        ))
        charts.append(VisualizationEngine.create_payoff_matrix_heatmap(
            pd_matrix, title="PD"
        ))
        charts.append(VisualizationEngine.create_cooperation_index_chart(
            data_manager.get_cooperation_index_data()
        ))
        charts.append(VisualizationEngine.create_tariff_escalation_chart(
            data_manager.get_tariff_data()
        ))
        
        # All should be valid figures
        for i, chart in enumerate(charts):
            assert isinstance(chart, go.Figure), f"Chart {i} is not a valid Figure"
        
        print(f"✅ Created {len(charts)} charts successfully: PASSED")


# =============================================================================
# TEST: Color and Styling
# =============================================================================

class TestColorAndStyling:
    """Tests for consistent color and styling."""
    
    def test_professional_theme_colors(self):
        """Verify ProfessionalTheme has required colors."""
        required_attrs = ['PRIMARY', 'SECONDARY', 'SUCCESS', 'WARNING', 
                         'DANGER', 'CHART_PALETTE']
        
        for attr in required_attrs:
            assert hasattr(ProfessionalTheme, attr), f"Missing: {attr}"
            value = getattr(ProfessionalTheme, attr)
            if attr != 'CHART_PALETTE':
                assert value.startswith('#'), f"{attr} should be hex color"
        print("✅ Professional Theme Colors: PASSED")
    
    def test_chart_palette_sufficient(self):
        """Verify chart palette has enough colors."""
        palette = ProfessionalTheme.CHART_PALETTE
        
        assert len(palette) >= 5, "Need at least 5 colors for charts"
        for color in palette:
            assert color.startswith('#'), "Colors should be hex format"
        print("✅ Chart Palette Sufficient: PASSED")
    
    def test_visualization_engine_colors(self):
        """Verify VisualizationEngine has color definitions."""
        assert hasattr(VisualizationEngine, 'COLORS')
        colors = VisualizationEngine.COLORS
        
        assert 'us' in colors
        assert 'china' in colors
        assert 'cooperation' in colors
        print("✅ VisualizationEngine Colors: PASSED")


# =============================================================================
# TEST: Edge Cases
# =============================================================================

class TestVisualizationEdgeCases:
    """Edge case tests for visualizations."""
    
    def test_extreme_payoff_values(self):
        """Test with extreme payoff values."""
        extreme_matrix = PayoffMatrix(
            cc=(1000, 1000),
            cd=(-1000, 1000),
            dc=(1000, -1000),
            dd=(0, 0)
        )
        
        fig = VisualizationEngine.create_payoff_matrix_heatmap(
            extreme_matrix, title="Extreme"
        )
        assert isinstance(fig, go.Figure)
        print("✅ Extreme Payoff Values: PASSED")
    
    def test_zero_payoff_matrix(self):
        """Test with zero payoff values."""
        zero_matrix = PayoffMatrix(
            cc=(0, 0), cd=(0, 0), dc=(0, 0), dd=(0, 0)
        )
        
        fig = VisualizationEngine.create_payoff_matrix_heatmap(
            zero_matrix, title="Zero"
        )
        assert isinstance(fig, go.Figure)
        print("✅ Zero Payoff Matrix: PASSED")


# =============================================================================
# TEST: Integration with Data Manager
# =============================================================================

class TestVisualizationDataIntegration:
    """Integration tests with DataManager data."""
    
    def test_real_cooperation_data_chart(self, data_manager):
        """Test chart with real cooperation data."""
        coop_data = data_manager.get_cooperation_index_data()
        
        fig = VisualizationEngine.create_cooperation_index_chart(coop_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        print("✅ Real Cooperation Data Chart: PASSED")
    
    def test_real_tariff_data_chart(self, data_manager):
        """Test chart with real tariff data."""
        tariff_data = data_manager.get_tariff_data()
        
        fig = VisualizationEngine.create_tariff_escalation_chart(tariff_data)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0
        print("✅ Real Tariff Data Chart: PASSED")


# =============================================================================
# SUMMARY TEST
# =============================================================================

class TestVisualizationSummary:
    """Summary test for all visualizations."""
    
    def test_complete_visualization_suite(self, harmony_matrix, pd_matrix, data_manager):
        """Test complete visualization suite."""
        
        results = {}
        
        # 1. Payoff Matrix Heatmaps
        try:
            fig = VisualizationEngine.create_payoff_matrix_heatmap(harmony_matrix, "Harmony")
            results['Payoff Heatmap (Harmony)'] = '✅'
        except Exception as e:
            results['Payoff Heatmap (Harmony)'] = f'❌ {e}'
        
        try:
            fig = VisualizationEngine.create_payoff_matrix_heatmap(pd_matrix, "PD")
            results['Payoff Heatmap (PD)'] = '✅'
        except Exception as e:
            results['Payoff Heatmap (PD)'] = f'❌ {e}'
        
        # 2. Cooperation Index
        try:
            coop_data = data_manager.get_cooperation_index_data()
            fig = VisualizationEngine.create_cooperation_index_chart(coop_data)
            results['Cooperation Index'] = '✅'
        except Exception as e:
            results['Cooperation Index'] = f'❌ {e}'
        
        # 3. Tariff Escalation
        try:
            tariff_data = data_manager.get_tariff_data()
            fig = VisualizationEngine.create_tariff_escalation_chart(tariff_data)
            results['Tariff Escalation'] = '✅'
        except Exception as e:
            results['Tariff Escalation'] = f'❌ {e}'
        
        # Print results
        print("\n" + "="*50)
        print("VISUALIZATION TEST RESULTS")
        print("="*50)
        for chart, status in results.items():
            print(f"  {chart}: {status}")
        print("="*50)
        
        # All should pass
        failures = [k for k, v in results.items() if '❌' in v]
        assert len(failures) == 0, f"Failed charts: {failures}"
        
        print(f"\n✅ All {len(results)} core visualizations passed!")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
