"""LaTeX -> terminal text conversion used by the chat box (display only)."""
from alice.render import latex_to_text


def test_display_maths_becomes_readable_text():
    src = r"\[ \text{lrate} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot w^{-1.5}) \]"
    assert latex_to_text(src) == "lrate = d_model^(-0.5) · min(step^(-0.5), step · w^(-1.5))"


def test_inline_maths_symbols_scripts_and_fractions():
    assert latex_to_text(r"so $x^2 + y_1^2 = r^2$ here") == "so x² + y₁² = r² here"
    assert latex_to_text(r"\(\beta_1 = 0.9,\ \epsilon = 10^{-9}\)") == "β₁ = 0.9, ε = 10⁻⁹"
    assert latex_to_text(r"$$\frac{a+b}{2} \approx \sqrt{x}$$") == "(a+b)/(2) ≈ √(x)"
    assert latex_to_text(r"$P(D \mid +) \le 1$") == "P(D | +) ≤ 1"


def test_emoji_become_ascii_emoticons_but_symbols_and_code_stay():
    from alice.render import ascii_emoji
    assert ascii_emoji("Nice work 😄 and well done 👍") == "Nice work :D and well done (y)"
    assert ascii_emoji("hmm 🤔️ really") == "hmm :/ really"                      # with a variation selector
    assert ascii_emoji("party 🎉 time") == "party  time"                         # no natural emoticon: dropped
    assert ascii_emoji("x → y, ⌈n/2⌉, done ✓") == "x → y, ⌈n/2⌉, done ✓"       # technical symbols kept
    code = "```python\nprint('🙂')\n    return x\n```"
    assert ascii_emoji(code) == code
    assert ascii_emoji("    indented") == "    indented"                         # streamed code keeps its spaces


def test_code_money_and_plain_text_are_left_alone():
    assert latex_to_text("costs $5 and $10 per run") == "costs $5 and $10 per run"
    code = "```bash\necho $HOME and $x$\n```"
    assert latex_to_text(code) == code
    assert latex_to_text("use `$PATH` then $a$") == "use `$PATH` then a"
    assert latex_to_text("β₁ = 0.9, no maths markup") == "β₁ = 0.9, no maths markup"
