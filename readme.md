# **Is It Really Appropriate To Be So Ironic?**

## **1. Obiettivo**

Questo lavoro esplora **quando l’uso dell’ironia è considerato appropriato** all’interno di una conversazione.

In particolare, ci concentriamo su **dialoghi estratti dai social media**, con l’obiettivo di capire **se e come i modelli linguistici possano apprendere** quando usare l’ironia, in modo **contestuale** e **sensibile alla prospettiva del parlante**.

---

## **2. Assunzione di base**

Assumiamo che:

> **Se, in una conversazione umana, è stata utilizzata una risposta ironica, allora essa era (per il parlante umano) appropriata.**

Questa assunzione ci permette di utilizzare il comportamento umano osservato come proxy dell’“appropriatezza”.

---

## **3. Datasets**

Utilizziamo due risorse complementari:

| Dataset       | Contenuto                                                             | Limiti                         | Uso nel nostro lavoro                                    |
| ------------- | --------------------------------------------------------------------- | ------------------------------ | -------------------------------------------------------- |
| **MultiPICo** | Coppie post→reply annotate con ironia, con annotazioni prospettiviste | Contesto conversazionale breve | Estendiamo il **contesto**                               |
| **CSC**       | Dialoghi multi-turn                  | Mancano info socio-demografiche  | Lo **arricchiamo con annotazioni prospettiviste silver** |

---

## **4. Metodologia**

La pipeline è organizzata in tre fasi:

### **4.1 Data Augmentation**

* **Estensione del contesto in MultiPICo**: utilizziamo LLM per generare conversazioni più ampie per passare da coppie post→reply a **dialoghi multi-turn**.
* **Annotazione prospettivista in CSC**: alleniamo un classificatore dell’ironia su MultiPICo e lo utilizziamo per produrre **annotazioni silver** del CSC, preservando le informazioni sulla prospettiva.


### **4.2 Model Evaluation**

L’obiettivo è verificare se i modelli sono in grado di **selezionare** (o **produrre**) una risposta ironica quando è appropriato.

**Dubbio metodologico chiave:**
Trattiamo il problema come **classificazione** o come **generazione**?

| Approccio           | Descrizione                                                                                  | Pro                                              | Contro                                                                                   |
| ------------------- | -------------------------------------------------------------------------------------------- | ------------------------------------------------ | ---------------------------------------------------------------------------------------- |
| **Classificazione** | Dato il contesto e la prospettiva, il modello predice se la risposta dovrebbe essere ironica (0/1) | Valutazione chiara (accuracy/F1), setup semplice | Minor ricchezza analitica                                                                |
| **Generazione**     | Il modello genera direttamente la reply                                                  | Permette analisi qualitative sulle generazioni prodotte   | Richiede valutazione automatica complessa (es. classificatore o annotazione umana dell'ironia) |

---

### **4.3 Analisi**

Le analisi si concentrano su:

1. **Prospettiva e variazione nell’uso dell’ironia**
   Es.: il modello riproduce alcune prospettive più di altre?

2. **Adattamento tramite fine-tuning**
   Possiamo “orientare” un modello verso una particolare prospettiva riguardo all'appropriatezza dell'ironia?

---

## **5. Contributi attesi**

* Introduzione di versioni **arricchite** di MultiPICo e CSC (con contesto + annotazioni prospettiviste)
* Valutazione sistematica della capacità dei LLM di usare l’ironia **in modo contestualmente appropriato**
* Discussione sull’impatto della **prospettiva** nella generazione di contenuto ironico

---

## **6. Risorse**

* MultiPICo: [https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo)
* CSC: [https://github.com/CoPsyN/CSC](https://github.com/CoPsyN/CSC)
* Letteratura sulla prospettiva linguistica e l'ironia → (inserire paper più specifici nella versione finale)

---

Punto chiave: definire il convetto di **appropriatezza dell'ironia**