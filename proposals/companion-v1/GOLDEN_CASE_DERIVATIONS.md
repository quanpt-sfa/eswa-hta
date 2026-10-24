# Mathematical Derivations of Golden Test Cases

**Document:** `proposals/companion-v1/GOLDEN_CASE_DERIVATIONS.md`  
**Status:** PROPOSAL_CANDIDATE (Pending independent human sign-off)  
**Governing Specification:** `docs/HTA_IMPLEMENTATION_SPEC_v1.2.md` (§15.2)  

---

## 1. Overview and Purpose

This document provides independent algebraic derivations and verification proofs for every synthetic case in `golden_cases.csv` and `golden_expectations.csv`. All test cases are designed to probe edge cases, boundary conditions, and potential implementation errors without referencing empirical financial data.

---

## 2. Derivations by Test ID

### 2.1 Label Construction and Materiality Invariants

#### T-L01: Sub-threshold Positive Adjustment
- **Inputs:** $\text{PAT}_{\text{pre}} = 100$, $\text{PAT}_{\text{post}} = 104$, $k = 0.05$.
- **Derivation:**
  $$\Delta = 104 - 100 = 4$$
  $$k \cdot |\text{PAT}_{\text{pre}}| = 0.05 \times 100 = 5.0$$
  $$|\Delta| = 4 < 5.0 \implies y = 0$$
- **Eligibility:** $\text{PAT}_{\text{pre}} > 0, \text{PAT}_{\text{post}} > 0, \text{PAT}_{\text{pre}} \cdot \text{PAT}_{\text{post}} > 0 \implies \text{eligible\_label} = \text{True}$.
- **Expected Label:** $y = 0$, $\text{is\_zero\_adj} = \text{False}$.

#### T-L02: Inclusive Boundary Positive Adjustment
- **Inputs:** $\text{PAT}_{\text{pre}} = 100$, $\text{PAT}_{\text{post}} = 105$, $k = 0.05$.
- **Derivation:**
  $$\Delta = 105 - 100 = 5.0$$
  $$|\Delta| = 5.0 \ge 5.0 \implies y = 1$$
- **Rule Tested:** Inclusive inequality ($\ge$) at the exact threshold boundary.
- **Expected Label:** $y = 1$, $\text{is\_zero\_adj} = \text{False}$.

#### T-L03: Negative Adjustment at Threshold
- **Inputs:** $\text{PAT}_{\text{pre}} = 100$, $\text{PAT}_{\text{post}} = 95$, $k = 0.05$.
- **Derivation:**
  $$\Delta = 95 - 100 = -5.0 \implies |\Delta| = 5.0 \ge 5.0 \implies y = 1$$
- **Expected Label:** $y = 1$, $\text{is\_zero\_adj} = \text{False}$.

#### T-L04: Zero Pre-Audit Earnings (Denominator Singularity)
- **Inputs:** $\text{PAT}_{\text{pre}} = 0$, $\text{PAT}_{\text{post}} = 10$, $k = 0.05$.
- **Derivation:** Division by zero in relative change formula $|\Delta| / |\text{PAT}_{\text{pre}}|$.
- **Eligibility Contract (§2, D04):** $\text{PAT}_{\text{pre}} \neq 0$ is required for $\text{eligible\_label}$.
- **Expected Disposition:** $\text{eligible\_label} = \text{False}, \text{eligible\_model} = \text{False}$. Label $y$ is undefined/missing.

#### T-L05: Earnings Sign Crossing
- **Inputs:** $\text{PAT}_{\text{pre}} = 100$, $\text{PAT}_{\text{post}} = -100$, $k = 0.05$.
- **Derivation:** $\text{PAT}_{\text{pre}} \cdot \text{PAT}_{\text{post}} = -10,000 < 0$.
- **Eligibility Contract (§2, D04–D05):** Sign change cases are excluded from the primary sample.
- **Expected Disposition:** $\text{eligible\_label} = \text{False}, \text{eligible\_model} = \text{False}$.

#### T-L06: Post-Audit Earnings Collapse to Zero
- **Inputs:** $\text{PAT}_{\text{pre}} = 100$, $\text{PAT}_{\text{post}} = 0$, $k = 0.05$.
- **Derivation:** $\text{PAT}_{\text{pre}} \cdot \text{PAT}_{\text{post}} = 0 \ge 0$.
- **Rule (§2, D04):** $\text{PAT}_{\text{post}} = 0$ is permitted and eligible.
  $$\Delta = -100 \implies |\Delta| = 100 \ge 5.0 \implies y = 1$$
- **Expected Disposition:** $\text{eligible\_label} = \text{True}, \text{eligible\_model} = \text{True}, y = 1$.

#### T-L07 & T-L08: Negative Pre-Audit Earnings
- **Inputs (T-L07):** $\text{PAT}_{\text{pre}} = -100$, $\text{PAT}_{\text{post}} = -105$, $k = 0.05$.
  $$\Delta = -105 - (-100) = -5.0 \implies |\Delta| = 5.0 \ge 0.05 \times |-100| = 5.0 \implies y = 1$$
  $$\text{PAT}_{\text{pre}} \cdot \text{PAT}_{\text{post}} = (-100) \times (-105) = 10500 > 0 \implies \text{eligible\_label} = \text{True}$$
- **Inputs (T-L08):** $\text{PAT}_{\text{pre}} = -100$, $\text{PAT}_{\text{post}} = -95$, $k = 0.05$.
  $$\Delta = -95 - (-100) = +5.0 \implies |\Delta| = 5.0 \ge 5.0 \implies y = 1$$
- **Inputs (T-L09):** $\text{PAT}_{\text{pre}} = -100$, $\text{PAT}_{\text{post}} = -96$, $k = 0.05$.
  $$\Delta = +4.0 \implies |\Delta| = 4.0 < 5.0 \implies y = 0$$

---

### 2.2 Zero and Near-Zero Adjustment Invariants

#### T-Z01: Exact Zero Adjustment
- **Inputs:** $\text{PAT}_{\text{pre}} = 100$, $\text{PAT}_{\text{post}} = 100$, $k = 0.05$.
- **Derivation:** $\Delta = 0.0 \implies \text{is\_zero\_adj} = \text{True}, y = 0$.

#### T-Z02: Small Adjustment Below Materiality Threshold (CORRECTION)
- **Previous Specification Bug:** Prior candidate text listed expected $y = 1$ for this case.
- **Inputs:** $\text{PAT}_{\text{pre}} = 100$, $\text{PAT}_{\text{post}} = 100.0000000001$, $k = 0.05$.
- **Independent Derivation:**
  $$\Delta = 100.0000000001 - 100 = 10^{-10}$$
  $$\text{Materiality threshold: } 0.05 \times 100 = 5.0$$
  $$|\Delta| = 10^{-10} \ll 5.0 \implies y = 0$$
  $$\Delta \neq 0 \implies \text{is\_zero\_adj} = \text{False}$$
- **Corrected Expected Disposition:** $\text{is\_zero\_adj} = \text{False}$, $y = 0$.

#### T-Z03: Near-Zero Earnings with Genuine Threshold Exceedance (NEW)
- **Inputs:** $\text{PAT}_{\text{pre}} = 10^{-9}$, $\text{PAT}_{\text{post}} = 1.1 \times 10^{-9}$, $k = 0.05$.
- **Independent Derivation:**
  $$\Delta = 1.1 \times 10^{-9} - 1.0 \times 10^{-9} = 10^{-10}$$
  $$\text{Materiality threshold: } 0.05 \times 10^{-9} = 5.0 \times 10^{-11}$$
  $$|\Delta| = 10^{-10} = 10.0 \times 10^{-11} \ge 5.0 \times 10^{-11} \implies y = 1$$
  $$\Delta \neq 0 \implies \text{is\_zero\_adj} = \text{False}$$
- **Expected Disposition:** $\text{eligible\_label} = \text{True}$, $\text{is\_zero\_adj} = \text{False}$, $y = 1$.

---

### 2.3 Post-Audit Denominator Asymmetric Cuts

When sensitivity analyses define adjustments relative to post-audit earnings:
$$| \Delta / \text{PAT}_{\text{post}} | \ge k \iff |\Delta| \ge k \cdot |\text{PAT}_{\text{pre}} + \Delta|$$

#### Algebraic Derivation:
Let $k \in (0, 1)$. Assume no sign crossing ($\text{PAT}_{\text{pre}} \cdot \text{PAT}_{\text{post}} > 0$).

**Case A: $\text{PAT}_{\text{pre}} > 0$**
1. **Positive Adjustment ($\Delta > 0$):**
   $$\Delta \ge k(\text{PAT}_{\text{pre}} + \Delta) \iff \Delta(1 - k) \ge k \cdot \text{PAT}_{\text{pre}} \iff \Delta \ge \frac{k}{1 - k} \text{PAT}_{\text{pre}}$$
   For $\text{PAT}_{\text{pre}} = 100, k = 0.05$:
   $$\Delta_{\text{upper}} \ge \frac{0.05}{0.95} \times 100 = \frac{100}{19} \approx +5.263157894736842$$
2. **Negative Adjustment ($\Delta < 0$):**
   Since $\text{PAT}_{\text{post}} > 0$, we have $|\Delta| = -\Delta$ and $\text{PAT}_{\text{post}} = \text{PAT}_{\text{pre}} + \Delta > 0$:
   $$-\Delta \ge k(\text{PAT}_{\text{pre}} + \Delta) \iff -\Delta(1 + k) \ge k \cdot \text{PAT}_{\text{pre}} \iff \Delta \le -\frac{k}{1 + k} \text{PAT}_{\text{pre}}$$
   For $\text{PAT}_{\text{pre}} = 100, k = 0.05$:
   $$\Delta_{\text{lower}} \le -\frac{0.05}{1.05} \times 100 = -\frac{100}{21} \approx -4.761904761904762$$

**Case B: $\text{PAT}_{\text{pre}} < 0$**
Let $\text{PAT}_{\text{pre}} = -M$ with $M > 0$. Since $\text{PAT}_{\text{post}} < 0$, we have $|\text{PAT}_{\text{post}}| = M - \Delta > 0$.
1. **Upward Adjustment ($\Delta > 0$):**
   $$\Delta \ge k(M - \Delta) \iff \Delta(1 + k) \ge k M \iff \Delta \ge \frac{k}{1 + k} M = -\frac{k}{1 + k} \text{PAT}_{\text{pre}}$$
   For $\text{PAT}_{\text{pre}} = -100, k = 0.05$:
   $$\Delta_{\text{upper}} \ge \frac{0.05}{1.05} \times 100 = \frac{100}{21} \approx +4.761904761904762$$
2. **Downward Adjustment ($\Delta < 0$):**
   $$-\Delta \ge k(M - \Delta) \iff -\Delta(1 - k) \ge k M \iff \Delta \le -\frac{k}{1 - k} M = \frac{k}{1 - k} \text{PAT}_{\text{pre}}$$
   For $\text{PAT}_{\text{pre}} = -100, k = 0.05$:
   $$\Delta_{\text{lower}} \le -\frac{0.05}{0.95} \times 100 = -\frac{100}{19} \approx -5.263157894736842$$

**Verification Summary:**
- Positive $\text{PAT}_{\text{pre}}$: Lower cut is $\approx -4.7619$; Upper cut is $\approx +5.2632$.
- Negative $\text{PAT}_{\text{pre}}$: Lower cut is $\approx -5.2632$; Upper cut is $\approx +4.7619$.
- Cut magnitude reversal is mathematically proven.

---

### 2.4 Probability Exceedance and Hurdle Mixture

#### T-C01: Standard Normal Exceedance
- **Distribution:** $D \sim \mathcal{N}(0, 1)$, threshold $c = 1.0$.
- **Two-tailed Exceedance:**
  $$P(|D| \ge 1.0) = P(D \ge 1.0) + P(D \le -1.0) = 2 \cdot (1 - \Phi(1.0))$$
  $$\Phi(1.0) \approx 0.8413447460685429 \implies P(|D| \ge 1.0) \approx 0.31731050786291415$$

#### T-C02: Standard Laplace Exceedance
- **Distribution:** $D \sim \text{Laplace}(0, 1)$, threshold $c = 1.0$.
- **Two-tailed Exceedance:**
  $$P(|D| \ge 1.0) = 2 \cdot \frac{1}{2} e^{-1.0} = e^{-1} \approx 0.36787944117144233$$

#### T-H01: Hurdle Mixture Exceedance
- **Inputs:** $\pi_0 = 0.4$, $D_{\text{nonzero}} \sim \mathcal{N}(0, 1)$, $c = 1.0$.
- **Hurdle Formula:**
  $$P(|D| \ge c) = (1 - \pi_0) \cdot P(|D_{\text{nonzero}}| \ge c) = (1 - 0.4) \times 0.31731050786291415 = 0.19038630471774849$$
