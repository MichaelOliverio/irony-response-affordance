# Setting Iniziale

Annotatori:
- Adam
- Gianfranco
- Javier
- Marino

Datasets
- CANTRIP_CSC
- CANTRIP_MultiPICo

=== CSC (93 istanze) ===
Fleiss kappa tra tutti gli annotatori: 0.3561711385061639

=== MultiPICo (86 istanze) ===
Fleiss kappa tra tutti gli annotatori: 0.3034348282585871

=== CSC and MultiPICo (179 istanze) ===
Fleiss kappa tra tutti gli annotatori: 0.34334363409957025

------------------------------------------------------------------------------------

Fleiss Kappa a gruppi di 3
========================================
Sheet: CSC
========================================

  Trio: Adam / Gianfranco / Javier
  Items comuni: 93
  Fleiss kappa: 0.2687

  Trio: Adam / Gianfranco / Marino
  Items comuni: 93
  Fleiss kappa: 0.3259

  Trio: Adam / Javier / Marino
  Items comuni: 93
  Fleiss kappa: 0.5550 --> Migliore

  Trio: Gianfranco / Javier / Marino
  Items comuni: 93
  Fleiss kappa: 0.2684

========================================
Sheet: MultiPICo
========================================

  Trio: Adam / Gianfranco / Javier
  Items comuni: 86
  Fleiss kappa: 0.2952

  Trio: Adam / Gianfranco / Marino
  Items comuni: 86
  Fleiss kappa: 0.2584

  Trio: Adam / Javier / Marino
  Items comuni: 86
  Fleiss kappa: 0.3743 --> Migliore

  Trio: Gianfranco / Javier / Marino
  Items comuni: 86
  Fleiss kappa: 0.2807


Rimuoviamo l'annotatore con più disagreement: Gianfranco

# Setting senza Gianfranco

=== CSC (94 istanze) ===
Fleiss kappa tra tutti gli annotatori: 0.5582706766917295

=== MultiPICo (86 istanze) ===
Fleiss kappa tra tutti gli annotatori: 0.37430709534368084

=== CSC and MultiPICo (180 istanze) ===
Fleiss kappa tra tutti gli annotatori: 0.4794138585666823

========================================
Sheet: CSC
========================================

  Trio: Adam / Javier / Marino
  Items comuni: 94
  Fleiss kappa: 0.5583

========================================
Sheet: MultiPICo
========================================

  Trio: Adam / Javier / Marino
  Items comuni: 86
  Fleiss kappa: 0.3743


Note:
- dataset = labeling ottenuto tramite working hypotesis
- gold = labeling Amazon

============================================================
CONFRONTO REALE vs RANDOM(p) BASELINE
============================================================

────────────────────────────────────────────────────────────
  Global — tutti gli item (n=180)
  prevalenza dataset=49.4%  prevalenza gold=31.7%
────────────────────────────────────────────────────────────
  Metrica                 Reale      Random(p)          Δ
  ──────────────────   ──────────   ────────────   ────────
  Accuracy                0.600   0.501±0.038   +0.099
  Precision (1)           0.416   0.316±0.036   +0.100
  Recall (1)              0.649   0.494±0.068   +0.155
  F1 (1)                  0.507   0.385±0.045   +0.122
  Cohen κ                 0.197   -0.001±0.071   +0.198
  Spearman ρ              0.211   -0.001±0.076   +0.212

────────────────────────────────────────────────────────────
  CSC — tutti gli item (n=81)
  prevalenza dataset=55.6%  prevalenza gold=32.1%
────────────────────────────────────────────────────────────
  Metrica                 Reale      Random(p)          Δ
  ──────────────────   ──────────   ────────────   ────────
  Accuracy                0.568   0.481±0.056   +0.087
  Precision (1)           0.400   0.322±0.048   +0.078
  Recall (1)              0.692   0.558±0.098   +0.135
  F1 (1)                  0.507   0.407±0.062   +0.100
  Cohen κ                 0.169   0.002±0.101   +0.166
  Spearman ρ              0.189   0.003±0.114   +0.187

────────────────────────────────────────────────────────────
  MultiPICo — tutti gli item (n=65)
  prevalenza dataset=44.6%  prevalenza gold=36.9%
────────────────────────────────────────────────────────────
  Metrica                 Reale      Random(p)          Δ
  ──────────────────   ──────────   ────────────   ────────
  Accuracy                0.646   0.511±0.060   +0.135
  Precision (1)           0.517   0.367±0.067   +0.151
  Recall (1)              0.625   0.442±0.097   +0.183
  F1 (1)                  0.566   0.399±0.075   +0.167
  Cohen κ                 0.272   -0.005±0.119   +0.277
  Spearman ρ              0.275   -0.006±0.121   +0.281

============================================================
CONFRONTO REALE vs RANDOM(p) BASELINE — solo unanimi
============================================================

────────────────────────────────────────────────────────────
  Global unanimi (n=115)
  prevalenza dataset=44.3%  prevalenza gold=32.2%
────────────────────────────────────────────────────────────
  Metrica                 Reale      Random(p)          Δ
  ──────────────────   ──────────   ────────────   ────────
  Accuracy                0.652   0.520±0.046   +0.132
  Precision (1)           0.471   0.321±0.050   +0.149
  Recall (1)              0.649   0.443±0.084   +0.205
  F1 (1)                  0.545   0.372±0.061   +0.174
  Cohen κ                 0.275   -0.000±0.092   +0.275
  Spearman ρ              0.284   -0.000±0.095   +0.285

────────────────────────────────────────────────────────────
  CSC unanimi (n=47)
  prevalenza dataset=53.2%  prevalenza gold=29.8%
────────────────────────────────────────────────────────────
  Metrica                 Reale      Random(p)          Δ
  ──────────────────   ──────────   ────────────   ────────
  Accuracy                0.596   0.486±0.071   +0.110
  Precision (1)           0.400   0.297±0.061   +0.103
  Recall (1)              0.714   0.531±0.128   +0.183
  F1 (1)                  0.513   0.379±0.080   +0.133
  Cohen κ                 0.212   -0.002±0.126   +0.214
  Spearman ρ              0.238   -0.002±0.143   +0.240

────────────────────────────────────────────────────────────
  MultiPICo unanimi (n=47)
  prevalenza dataset=36.2%  prevalenza gold=38.3%
────────────────────────────────────────────────────────────
  Metrica                 Reale      Random(p)          Δ
  ──────────────────   ──────────   ────────────   ────────
  Accuracy                0.723   0.534±0.070   +0.189
  Precision (1)           0.647   0.385±0.099   +0.262
  Recall (1)              0.611   0.362±0.113   +0.249
  F1 (1)                  0.629   0.370±0.099   +0.259
  Cohen κ                 0.409   0.003±0.147   +0.405
  Spearman ρ              0.409   0.003±0.149   +0.406
