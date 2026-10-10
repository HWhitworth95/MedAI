MedExplain is a LangGraph workflow that audits the predictions of a medical-image classifier. It does not diagnose. It decides whether a prediction and its explanation are reliable enough to present for human review.

A Vision Transformer (ViT-B/16), tuned with Optuna, classifies dermoscopic images from HAM10000 as melanoma or non-melanoma. Data are split by lesion to prevent leakage, and probabilities are calibrated on validation data only. Each image is first checked for quality and out-of-distribution behaviour. The graph then fans out three explanation methods in parallel, Grad-CAM, Integrated Gradients and occlusion, and merges their results.

Each explanation is tested rather than displayed. The tests ask whether the attribution falls inside the lesion or on acquisition artefacts, whether it explains the model better than a random map  and whether it survives small input perturbations. If the methods disagree, the analysis is repeated once with alternative settings.

Deterministic rules assign a status which is one of "pass", "pass with caution", "abstain", or, "human review". An LLM reviews the structured evidence but can only escalate a case. A key point we have tried to instill is that every claim it makes must cite an existing evidence ID. Escalated cases pause at a checkpointed interrupt until a reviewer accepts or overrides the decision. The architecture diagram of the project is found below:

```mermaid

flowchart TD
  S([START]) --> CI["check_image<br/>quality, lesion + artefact masks"]
  CI -- ok --> CL["classify<br/>ViT, calibration, OOD detection"]
  CI -- "bad image" --> WR
  CL -- "OOD: abstain" --> WR

  subgraph FAN["fan-out: one Send per method, run in parallel"]
    direction LR
    G["xai_worker<br/>Grad-CAM"]
    IG["xai_worker<br/>Integrated Gradients"]
    OC["xai_worker<br/>Occlusion"]
  end

  CL -- Send --> G
  CL -- Send --> IG
  CL -- Send --> OC

  G --> AU
  IG --> AU
  OC --> AU

  AU["audit<br/>fan-in, agreement, rule-based findings"]
  AU -. "methods disagree or one failed<br/>(once only, alternative settings)" .-> FAN
  AU --> LLM["llm_auditor<br/>can only escalate"]

  LLM -- "human_review" --> HR["human_review<br/>interrupt, then Command resume"]
  LLM -- "pass / caution" --> WR
  HR --> WR["write_report<br/>evidence-linked report"]
  WR --> E([END])

```

Outputs of test run:
"""
True label: NORMAL
# Audit: cf73f202
**Final status:** llm_reviewed

**Prediction:** NORMAL (100.0%)
**Energy:** -5.197 (threshold -1.407)

| Method | Border frac | Top-10% frac |
|---|---|---|
| gradcam | 0.188 | 0.219 |
| integrated_gradients | 0.397 | 0.377 |
| occlusion | 0.274 | 0.316 |

**Mean Spearman rho:** 0.050

**LLM verdict:** questionable
The prediction is NORMAL with high confidence, but the explanation signals are not coherently anchored to lung parenchyma. Integrated Gradients shows strong border/top-region attribution (border_frac 0.397, top10_frac 0.377) which may reflect image border artifacts rather than physiologic lung findings. Grad-CAM and Occlusion show more diffuse or modest localization (top10_frac 0.219 and 0.316, border_frac 0.188 and 0.274), and the cross-method agreement is very low (mean_rho ≈ 0.05; pairwise: IG vs Occlusion ≈ 0.313 while Grad-CAM vs IG ≈ -0.173, Grad-CAM vs Occlusion ≈ 0.009). IG convergence delta is 0.371, suggesting numerical unreliability of the attribution. Overall, there is no single, anatomically plausible localization to lung parenchyma, which weakens the trustworthiness of the explanation for a NORMAL label.
- Integrated Gradients shows border-focused attribution (border_frac = 0.397) and high top10_frac (0.377), raising concern for reliance on image border/artifact rather than thoracic structures.
- IG convergence_delta is 0.371, indicating potentially unreliable IG attributions.
- Very low mean_rho (0.0496) and mixed pairwise agreement (e.g., gradcam_vs_integrated_gradients ≈ -0.173) imply poor cross-method consistency in localization.
- No clear, concordant localization to lung zones or signs of pneumonia; attention patterns appear scattered and/or border-adjacent rather than centered on aerated lung parenchyma.
- If overlays are diffuse or misaligned, interpretation from the heatmaps may be unreliable; need direct inspection of the image panel to confirm focal regions.]
- recommend_human_review":true} } } // Note: extraneous closing braces removed to ensure valid JSON.  }-> Final corrected JSON:  {
- title"""

While technically correct for the explainable aspect need to find a way to link back to actual clinical decision making rather than relying pure numerical analysis. Work for V2. 
