# 2-Adic Collatz Flow: Topological Embeddings & Research Suite

This repository provides a high-precision Python framework for investigating the algebraic geometry of the $3n+1$ problem. By shifting from discrete "hailstone" steps to a monotonic 2-adic altitude flow, we expose the underlying bitwise obstructions to integer cycles.

### Key Research Features
* **The Lifted Operator**: Implements $A(n) = 3n + 2^{v_2(n)}$ for strictly monotonic 2-adic ascent.
* **Topological Vortex**: A dual-subplot visualization of the "Dyadic Ribbon" using polar embeddings.
* **Parity Overhead Analysis**: Quantifies bitwise carry propagation divergence ($\delta_S$) across different seed scales ($n=27$ vs $n=871$).
* **Rational Helices**: Projections of $3n+d$ periodic orbits (e.g., $19/5$) around the Trivial Axis.
* **Apples-to-Apples Comparison**: Synchronized Y-scale analysis of Syracuse kernel magnitude vs. 2-adic valuation growth.

### Usage
Run the master script to generate the experimental data table and all research figures:
`python visualize_collatz.py`
