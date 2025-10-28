# IS IT REALLY APPROPRIATE TO BE SO IRONIC?

Questo repository contiene dati, codice ed esperimenti relativi allo studio dell’appropriatezza dell’ironia nei dialoghi. L’obiettivo principale è esplorare quando è appropriato rispondere con ironia, nonchè il rilascio di una versione arricchita di MultiPICo (D-MultiPICo).

## Metodologia

La nostra metodologia è strutturata in tre fasi principali: **generazione del contesto**, **annotazione umana** e **analisi**.

### 1. Generazione del Contesto

Utilizziamo **MultiPICo** per generare un contesto più ampio attorno a post e risposte sui social media. In particolare:

- **Esempio di Post:**
```

Red rooster is the worst fast food place by far. Where I grew up there were two: one in the major shopping centre and one about a minute away outside. The outdoor one was always dead; never saw a car in the drive-thru. But the inside one had a line as long as Hungry Jack's. Almost like people not wanting to wait in line decided to go there, which then formed a line. There is a Chinese place, Maccas, KFC and a bunch of smaller food places in the same food court as well. But always a line at Red Rooster.

```
- **Esempio di Risposta:**
```

Worse than Maccas? Sure mate

```
- **Annotation Task:** È appropriato rispondere con una frase ironica? (Sì/No)

Bisogna chiarire quanto contesto dare per l'annotazione. Ci sono più strade
- solo il post
- post + reply
- 1+ turni precedenti generato + post + reply

Potrebbe essere anche interessante studiare quanto contesto serve per capire l'appropriatezza dell'ironia, quindi in tal caso verrebbero forniti agli annotatori dialoghi di diversa lunghezza

Il contesto aggiuntivo alle coppie post-reply verrà generato usando LLM.

### 2. Annotazione Umana

I dataset sintetici generati dagli LLM vengono poi annotati su **Prolific**. Gli annotatori valutano l’appropriatezza di una risposta ironica nel contesto fornito.

### 3. Analisi

Dopo l’annotazione, vengono condotte diverse analisi:

- **Analisi 1:** Come variano le annotazioni umane in base alla prospettiva?
- **Analisi 2:** Gli LLM hanno una certa prospettiva? si potrebbe fare una valutazione utilizzando zero-shot
- **Analisi 3:** Il fine-tuning di un LLM permette l’adattamento a una prospettiva specifica sull’ironia?

### Risorse & Riferimenti

- Dataset MultiPICo (tutte le lingue)
- Letteratura rilevante sull’ironia nei dialoghi e sul prospettivismo:
  - https://consensus.app/search/when-to-be-ironic-in-dialogue-systems/OJxdPJBzQj2ZlAerJsLCQw/
  - https://consensus.app/search/when-to-be-ironic/0CuAicniT7CE1eJwAEoDwg/

---

## Bozza struttura del Paper

Di seguito la struttura prevista per il paper sullo studio dell’appropriatezza dell’ironia nei dialoghi:

### 1. Introduction
- Teoria sull’appropriatezza e uso dell’ironia
- Prospettivismo e ironia

### 2. Related Work
- Prospettivismo e teorie correlate
- Lavori già svolti in questa direzione (TO DO)

### 3. Dataset
- MultiPICo (tutte le lingue)

### 4. Metodologia
- Creazione del contesto
- Annotazione su Prolific
- Analisi dei dati

### 5. Annotazione di D-MultiPICo
- Descrizione del processo di annotazione sintetica generata con LLM
- Strategie di confronto con annotazioni umane

### 6. Analisi
- Analisi della variazione delle annotazioni
- Analisi delle prospettive LLM zero-shot
- Effetti del fine-tuning su prospettive specifiche

### 7. Conclusioni
- Sintesi dei risultati
- Implicazioni per l’uso dell’ironia nei sistemi di dialogo
- Possibili sviluppi futuri