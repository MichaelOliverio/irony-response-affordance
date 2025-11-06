# **Teach Me To Joke**

## **1. Obiettivo**

Questo lavoro esplora **quando l’uso dell’ironia è considerato appropriato** in una conversazione.

---

## **2. Research Questions**

- **RQ1**: When is it appropriate to be ironic?
- **RQ2**: To whom is it appropriate to be ironic?

## **3. Working Hypothesis**

> **se, in un determinato momento del dialogo, è stata rilevata una risposta ironica o sarcastica da parte dello speaker o dell’observer, possiamo presumere che, per quello stesso ruolo, l’uso dell’ironia o del sarcasmo sia considerato appropriato.**

---

## **4. Datasets**

| Dataset       | Contenuto / Struttura                                                        | Informazioni sugli annotatori / Relazioni                                                                 | Tipo di dialogo        |
|---------------|-----------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|------------------------|
| **MultiPICo** | Irony labeling su conversazioni **single-turn** (post → reply)              | Include **informazioni socio-demografiche**; annotazione **disaggregata**; ogni conversazione annotata da **più annotatori esterni** | Single-turn            |
| **CSC**       | Sarcasm labeling su **contesto + reply**                                    | **Nessuna** informazione socio-demografica; annotazione da **speaker** e **observer** con relazione *best_friend / colleague*         | Contesto + reply       |


---

## **4. Metodologia**

La pipeline è organizzata in tre fasi:

### **4.1 Data Augmentation**

* **Estensione del contesto in MultiPICo**: utilizziamo LLM per generare conversazioni più ampie per passare da coppie post→reply/contesto→reply a **dialoghi multi-turn**.

  - Come aumentare il numero di turni?
    - letteratura utile:
      - [S2M: Converting Single-Turn to Multi-Turn Datasets for Conversational Question Answering (B. Li et al., 2023)](https://arxiv.org/html/2312.16511v1/#bib.bib14)
      - [SMILE: Single-turn to Multi-turn Inclusive Language Expansion via ChatGPT for Mental Health Support (Qiu et al., 2024)](https://aclanthology.org/2024.findings-emnlp.34.pdf?utm_source=chatgpt.com)
      - [Survey: Synthetic Data Generation Using Large Language Models: Advances in Text and Code (M. Nadas et al., 2025)](https://arxiv.org/abs/2503.14023)
      - [SPADE: Structured Prompting Augmentation for Dialogue Enhancement in Machine-Generated Text Detection (H. Li et al., 2024)](https://aclanthology.org/2025.llmsec-1.11.pdf)

  - LLM-as-a-Judjge per dialoghi sintetici
    - letteratura utile:
      - [Judging LLM-as-a-judge with MT-bench and Chatbot Arena](https://dl.acm.org/doi/10.5555/3666122.3668142)  

  - Quanti turni precedenti generare?
    - Letteratura utile:
      - [An Empirical Study on Context Length for Open-Domain Dialog Generation (X. Shen et al., 2024)](https://arxiv.org/html/2409.00315v1?utm_source=chatgpt.com)
      - [How Does Conversation Length Impact User’s Satisfaction? A Case Study of Length-Controlled Conversations with LLM-Powered Chatbots (S. Huang et al., 2024)](https://arxiv.org/html/2404.17025v1?utm_source=chatgpt.com)
      - [Context Does Matter: Implications for Crowdsourced Evaluation Labels in Task-Oriented Dialogue Systems (C. Siro et al., 2024)](https://www.researchgate.net/publication/381671868_Context_Does_Matter_Implications_for_Crowdsourced_Evaluation_Labels_in_Task-Oriented_Dialogue_Systems)


### **4.2 Model Evaluation**

L’obiettivo è verificare se i modelli sono in grado di **classificare** l'appropriatezza dell'uso dell'ironia/sarcasmo in un certo punto della conversazione

Metriche:
- Accuracy→con dati aggregati
- Precision/Recall→con dati disaggregati

---


## **5. Contributi attesi**

* Introduzione di versioni **arricchite** di MultiPICo e CSC (con contesto + annotazioni prospettiviste)
* Valutazione sistematica della capacità dei LLM di classificare quando **è appropriato** essere ironici/sarcastici
* Discussione sull’impatto della **prospettiva** nell'uso appropriato dell'ironia
---

## **6. Risorse**

* MultiPICo: [https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo)
* CSC: [https://github.com/CoPsyN/CSC](https://github.com/CoPsyN/CSC)
