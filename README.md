# numgrad

`numgrad` is a compact numerical-gradient and robustness demonstration built
with NumPy. It combines:

- analytical Ridge and binary cross-entropy gradients;
- central finite-difference gradient verification;
- memory-layout and contiguity inspection;
- a synthetic telemetry experiment showing how extreme poisoning affects
  standard Z-scores compared with robust MAD scores; and
- a four-panel Matplotlib diagnostic dashboard.

## Requirements

- Python 3.10 or newer
- NumPy
- Matplotlib

## Setup

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux, use `python3 -m venv .venv` and
`source .venv/bin/activate` instead.

## Run the demo

From the repository root:

```powershell
python run_pipeline.py
```

The demo uses deterministic synthetic data (random seed `42` for the
telemetry experiment) and prints:

- array metadata and contiguous versus transposed reduction timings;
- Ridge and BCE analytical-versus-numerical gradient checks; and
- anomaly counts before and after injecting extreme telemetry values.

It also writes `eda_diagnostics.png`, a four-panel dashboard in the repository
root. The generated image is ignored by Git.

## Run tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover both gradient implementations and the shape guard that
prevents accidental NumPy broadcasting.

## Project layout

```text
core_engine/
  losses.py       # Stable loss and analytical-gradient implementations
  memory.py       # Array metadata and contiguity benchmark
  robust_eda.py   # Synthetic poisoning/anomaly experiment
  verifier.py     # Central finite differences and error reporting
  visualizer.py   # Diagnostic dashboard rendering
run_pipeline.py   # End-to-end demo
tests/            # Unit tests
```