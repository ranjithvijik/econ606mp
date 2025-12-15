import pytest
from streamlit.testing.v1 import AppTest

# Define the proofs to test (Sample representative set)
PROOFS_TO_TEST = [
    ("1. Nash Equilibrium Analysis", "1.1 Nash Equilibrium Existence (Theorem 1.1)"),
    ("1. Nash Equilibrium Analysis", "1.3 Nash Equilibrium - Prisoner's Dilemma (Theorem 1.3)"),
    ("4. Folk Theorem & Repeated Games", "4.3 Tit-for-Tat Sustainability (Theorem 4.3)"),
    ("5. Discount Factor Thresholds", "5.1 Critical Discount Factor Formula (Theorem 5.1)"),
    ("5. Discount Factor Thresholds", "5.2 Cooperation Margin Formula (Theorem 5.2)"),
    ("9. Statistical Correlations", "9.2 Tariff Correlation Test (Theorem 9.2)")
]

def navigate_to_proofs_page(at):
    """Helper to navigate to the Mathematical Proofs page."""
    # 1. Select Main Category "📐 Mathematical Tools"
    nav_found = False
    for sb in at.sidebar.selectbox:
        if sb.label == "Select Research Area:":
            sb.set_value("📐 Mathematical Tools").run()
            nav_found = True
            break
    assert nav_found, "Main navigation selectbox not found"

    # 2. Select Page "📚 Mathematical Proofs"
    page_found = False
    for r in at.sidebar.radio:
        if r.label == "Go to Module:":
            r.set_value("📚 Mathematical Proofs").run()
            page_found = True
            break
    assert page_found, "Page navigation radio not found"

def test_proof_page_navigation():
    """Test navigating to the Mathematical Proofs page."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    navigate_to_proofs_page(at)
    
    # Verify page title or content loads (checking for page headline)
    # The headline is passed to render_mathematical_proofs_page but rendered as subheader/title
    # We look for "Mathematical Proofs & Derivations"
    found_title = any("Mathematical Proofs & Derivations" in ele.value for ele in at.markdown if hasattr(ele, 'value')) or \
                  any("Mathematical Proofs & Derivations" in ele.value for ele in at.title if hasattr(ele, 'value'))
                  
    assert found_title, "Page title 'Mathematical Proofs & Derivations' not found"
    assert not at.exception

def test_quick_access_buttons():
    """Test functionality of Quick Access Buttons."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    navigate_to_proofs_page(at)
    
    # Find button for Theorem 1.1
    target_label = "📐 1.1 Nash Equilibrium Existence (Theorem 1.1)"
    
    btn = next((b for b in at.button if b.label == target_label), None)
    assert btn, f"Quick access button for '{target_label}' not found"
    
    btn.click().run()
    
    # Verify Session State update
    assert at.session_state['selected_category'] == "1. Nash Equilibrium Analysis"
    assert at.session_state['selected_proof'] == "1.1 Nash Equilibrium Existence (Theorem 1.1)"
    
    # Verify content updated (Search for Theorem Title in Headers/Markdown)
    # "Theorem 1.1: Nash Equilibrium Existence"
    found_header = False
    all_text = []
    
    # Check markdown
    for md in at.markdown:
        all_text.append(md.value)
        if "Theorem 1.1: Nash Equilibrium Existence" in md.value:
            found_header = True
            break
            
    assert found_header, f"Proof content header not found. Found text: {all_text[:3]}"

@pytest.mark.parametrize("category, proof_name", PROOFS_TO_TEST)
def test_proof_content_rendering(category, proof_name):
    """Test that specific proofs render with valid LaTeX and formatting."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    navigate_to_proofs_page(at)
    
    # 1. Select Category
    cat_sb = next((sb for sb in at.selectbox if sb.key == "main_category_select"), None)
    assert cat_sb, f"Category Selectbox 'main_category_select' not found. Available keys: {[sb.key for sb in at.selectbox]}"
    cat_sb.set_value(category).run()
    
    # 2. Select Proof
    proof_sb = next((sb for sb in at.selectbox if sb.key == "main_proof_select"), None)
    assert proof_sb, "Proof Selectbox 'main_proof_select' not found"
    proof_sb.set_value(proof_name).run()
    
    # 3. Check content for LaTeX
    has_latex = False
    for md in at.markdown:
        if "$" in md.value:
            has_latex = True
            break
            
    assert has_latex, f"No LaTeX formulas ($ marker) found for {proof_name}"
    
    # Check specific content for 4.3 title
    if "4.3" in proof_name:
        found_target = any("0.40" in md.value for md in at.markdown)
        assert found_target, "Value '0.40' not found in Theorem 4.3 text"
