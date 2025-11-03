# **Is It Really Appropriate To Be So Ironic?**

## **1. Obiettivo**

Questo lavoro esplora **quando l’uso dell’ironia è considerato appropriato** all’interno di una conversazione.

In particolare, ci concentriamo su **dialoghi estratti dai social media**, con l’obiettivo di capire **se e come gli LLM possano apprendere** quando usare l’ironia, in modo **contestuale** e **sensibile alla prospettiva del parlante**.

---

## **2. Assunzione di base**

Assumiamo che:

> **Se, in una conversazione umana, è stata utilizzata una risposta ironica, allora essa era (per il parlante umano) appropriata.**

Questa assunzione ci permette di utilizzare il comportamento umano osservato come proxy dell’“appropriatezza”.

---

## **3. Datasets**

Utilizziamo due risorse complementari:

| Dataset       | Contenuto                                                             | Limiti                         | Estensione                                   |
| ------------- | --------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------- |
| **MultiPICo** | Coppie post→reply annotate con ironia, con annotazioni prospettiviste | Contesto conversazionale breve | Estendiamo il **contesto**                               |
| **CSC**       | Dialoghi multi-turn                  | Mancano info socio-demografiche  | Lo **arricchiamo con annotazioni prospettiviste silver** |

---

## **4. Metodologia**

La pipeline è organizzata in tre fasi:

### **4.1 Data Augmentation**

* **Estensione del contesto in MultiPICo**: utilizziamo LLM per generare conversazioni più ampie per passare da coppie post→reply a **dialoghi multi-turn**.

  - Come aumentare il numero di turni?
    - letteratura utile:
      - [S2M: Converting Single-Turn to Multi-Turn Datasets for Conversational Question Answering (B. Li et al., 2023)](https://arxiv.org/html/2312.16511v1/#bib.bib14)
      - [Survey: Synthetic Data Generation Using Large Language Models: Advances in Text and Code (M. Nadas et al., 2025)](https://arxiv.org/abs/2503.14023)
      - [SPADE: Structured Prompting Augmentation for Dialogue Enhancement in Machine-Generated Text Detection (H. Li et al., 2024)](https://aclanthology.org/2025.llmsec-1.11.pdf)

  - Quanti turni precedenti generare?
    - Letteratura utile:
      - [An Empirical Study on Context Length for Open-Domain Dialog Generation (X. Shen et al., 2024)](https://arxiv.org/html/2409.00315v1?utm_source=chatgpt.com)
      - [How Does Conversation Length Impact User’s Satisfaction? A Case Study of Length-Controlled Conversations with LLM-Powered Chatbots (S. Huang et al., 2024)](https://arxiv.org/html/2404.17025v1?utm_source=chatgpt.com)
      - [Context Does Matter: Implications for Crowdsourced Evaluation Labels in Task-Oriented Dialogue Systems (C. Siro et al., 2024)](https://www.researchgate.net/publication/381671868_Context_Does_Matter_Implications_for_Crowdsourced_Evaluation_Labels_in_Task-Oriented_Dialogue_Systems)

    - Nota personale: Io andrei a guradare CSC quanti turni ha mediamente e userei quello come numero di turni da generare per MultiPICo. Successivamente, andrei a valutare come incide per gli LLM il numero di turni dati in input sulle prestazioni


* **Annotazione prospettivista in CSC**: alleniamo un classificatore dell’ironia su MultiPICo e lo utilizziamo per produrre **annotazioni silver** del CSC, preservando le informazioni sulla prospettiva.


### **4.2 Model Evaluation**

L’obiettivo è verificare se i modelli sono in grado di **selezionare** (o **produrre**) una risposta ironica quando è appropriato.

**Dubbio metodologico chiave:**
Trattiamo il problema come **classificazione** o come **generazione**?

| Approccio           | Descrizione                                                                                  | Pro                                              | Contro                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Classificazione** | Dato il contesto e la prospettiva, il modello predice se la risposta dovrebbe essere ironica (0/1) | Valutazione chiara (accuracy/F1) | Minor ricchezza analitica                                                                |
| **Generazione**     | Il modello genera direttamente la reply                                                  | Permette analisi qualitative sulle generazioni prodotte   | Come valutare? (idea: fare classificazione automatica/manuale dell'ironia e poi comparazione con label gold) |

---

### **4.3 Analisi**

Le analisi si concentrano su:

1. **Prospettiva e variazione nell’uso dell’ironia**:
   Es. il modello riproduce alcune prospettive più di altre?

2. **Adattamento tramite fine-tuning**:
   Possiamo “orientare” un modello verso una particolare prospettiva riguardo all'appropriatezza dell'ironia?

---

## **5. Contributi attesi**

* Introduzione di versioni **arricchite** di MultiPICo e CSC (con contesto + annotazioni prospettiviste)
* Valutazione sistematica della capacità dei LLM di usare l’ironia **in modo contestualmente appropriato**
* Discussione sull’impatto della **prospettiva** nell'uso appropriato dell'ironia
---

## **6. Risorse**

* MultiPICo: [https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo)
* CSC: [https://github.com/CoPsyN/CSC](https://github.com/CoPsyN/CSC)

---

Punto chiave: definire il convetto di **appropriatezza dell'ironia**
