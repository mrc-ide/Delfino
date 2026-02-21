# Delfino: Trajectory-Based Epidemiological Simulation

Delfino is a simulation framework that uses Generative Pre-trained Transformers (GPTs) to model individual-level disease progression. It is designed to evaluate the long-term impact of clinical interventions (specifically GLP-1 receptor agonists) on obesity-related multi-morbidity.

## 🏛️ Scientific Foundation & Credits

The methodology of Delfino is an extension of the Delphi model contained in:
> **Shmatko, A. et al. (2024).** *Predicting future disease trajectories using transformers.* Nature Medicine.  
> [Read the full paper here](https://www.nature.com/articles/s41591-024-03102-1)

**Key Contributors to Delfino :**
* **Dr. Daniel Laydon** (Lead Developer)
* **Prof. Timothy Hallett**
* **Dr. Shevanthi Nayagam**
* **Prof. Alex Bottle**

## 🏗️ Technical Architecture

### 1. Modeling Paradigm
* **Sequential States:** Delfino treats a patient's life as a sequence of tokens. This allows the model to retain long-term dependencies in a patient's medical history.
* **Tokenization:** Clinical events, demographic markers (Sex, Age), and risk factors (BMI, Smoking) are mapped to a discrete vocabulary.
* **Inference:** The model predicts the "next likely clinical event" based on the entire preceding history, enabling the simulation of complex co-morbidity patterns.

### 2. Data & Mapping
* **Training Data:** Architecture is designed for training on longitudinal cohorts like the **UK Biobank** and **Whole Systems Integrated Care (WSIC)**.
* **Clinical Coding:** Model outputs are mapped to the **ICD-10** system (Chapters A00–Q99). This ensures that synthetic trajectories are expressed in standard clinical nomenclature (e.g., C-codes for oncology, I-codes for cardiovascular).
* **Explainable AI (XAI):** Uses **SHAP** (SHapley Additive exPlanations) values to calculate the contribution of historical tokens to specific future risk predictions, providing transparency for clinical validation.

### 3. Health Economics Engine
The engine calculates two primary health metrics:

* **DALYs (Disability-Adjusted Life Years):**
    * **YLD:** Accrued annually based on IHME/GBD disability weights mapped to ICD-10 tokens.
    * **YLL:** Calculated upon a "Death" token relative to actuarial life expectancy.
* **QALYs (Quality-Adjusted Life Years):**
    * **Additive:** Calculates quality decrements ($1 - utility$).
    * **Multiplicative:** Calculates composite utility ($U_{total} = \prod U_n$), reflecting standard HTA methodology for multi-morbid states.



## 💻 Implementation Details

### Performance & Scaling
* **Language:** Python/PyTorch.
* **Hardware:** Optimized for NVIDIA GPU architectures (CUDA).
* **Parallelization:** Uses a multi-process orchestrator to manage parallel worker streams, bypassing the Python GIL to achieve high throughput (measured in patients/second).
* **Efficiency:** Disease parameters (weights, costs, utilities) are vectorized into NumPy arrays for $O(1)$ lookup during the simulation loop.



### Execution Flow
1.  **Preprocessing:** `create_dummy_disease_data.py` maps labels to clinical weights.
2.  **Simulation:** `delfino.py` runs stochastic inference for Baseline and Intervention groups.
3.  **Orchestration:** `run_experiment.py` handles data slicing and subprocess management.
4.  **Analysis:** `compare_results.py` and `plot_results.py` generate incidence curves and cost-effectiveness metrics.