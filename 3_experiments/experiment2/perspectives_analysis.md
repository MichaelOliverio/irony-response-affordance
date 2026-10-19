# Neutral vs Conditioned: riferimento specifico fisso

## Riproduzione e file

Dalla root del repository, nell'ambiente con numpy, pandas, SciPy, scikit-learn,
statsmodels e Jinja2 (già specificati in `requirements.txt`):

```powershell
.venv/Scripts/python.exe -X utf8 3_experiments/experiment2/eval_perspectives.py
.venv/Scripts/python.exe -X utf8 3_experiments/experiment2/test_eval_perspectives.py -v
```

Non vengono caricati modelli, scaricate metriche o generate inferenze. Il nuovo
script non importa `src.config`, che caricherebbe torch: legge i percorsi canonici
documentati lì e riutilizza il metodo effettivo `_get_prompt` tramite AST, senza
inizializzare il decoder. Nessun file di predizioni o annotazioni viene modificato.

Output principale: `output/perspectives/`.

- `neutral_conditioned.csv`: tutti i 42 confronti, metriche, numerosità, esclusioni,
  copertura e diagnostica di protocollo.
- `traits.csv`, `traits.tex`: 36 confronti dei tratti, con MP e CSC affiancati
  nella tabella per il paper.
- `relationships.csv`, `relationships.tex`: sei confronti delle relazioni su CSC,
  con una riga per relazione entro ciascun blocco di modello.
- `traits_statistics.tex`, `relationships_statistics.tex`: tabelle estese con
  N, cluster, prevalenza, Neutral, Conditioned, Delta, IC e p-value.
- `pair_audit.csv.gz`: CSV compresso con una riga per ID e confronto, etichetta,
  entrambe le predizioni, cluster, presenza/validità, verifica del prompt,
  inclusione e motivi di esclusione. I motivi possono sovrapporsi.
- `input_manifest.json`: SHA-256 degli input e del codice usato, versioni
  dell'ambiente di valutazione, seed e numero di ricampionamenti.
- `complementary_global_perspective.csv` e `.tex`: confronto **descrittivo** delle
  stesse predizioni condizionate con Global e Perspective, sul campione appaiato
  della nuova analisi con entrambe le etichette valide. Il suo delta cambia
  riferimento e non misura il vantaggio del conditioning.

Le tabelle per il paper richiedono `booktabs` e `multirow`. Usano due decimali,
intestazioni `CANTRIP` MP/CSC, grassetto sul maggiore e su entrambe le celle in
caso di parità alla precisione mostrata. L'asterisco segue esclusivamente il
vincitore sui valori non arrotondati quando `p_value_holm < 0.05`, anche se
entrambi i valori arrotondati sono in grassetto. Il maggiore può essere Neutral.
Le caption indicano riferimento fisso, campione appaiato, test e famiglie Holm.
Le tabelle statistiche estese richiedono anche `longtable`; per le 13 colonne
usare una pagina orizzontale e un carattere piccolo.

La precedente analisi resta separata e invariata in `eval.py` e
`evaluation_results_semiaggregated.csv/json`. Anche `eval_llm_perspectives.py` e
i suoi risultati restano invariati. Nel repository non esisteva un file chiamato
`eval_perspectives.py`: è stato aggiunto come punto di ingresso della nuova
analisi. **I p-value storici di `eval.py` derivano dalle code del bootstrap
ordinario e non sono un test valido della nuova domanda.** Non vengono riutilizzati.

## Dati, abbinamento e proxy labels

Le inferenze principali sono i 48 CSV in `output/decoders/`, percorso usato da
`eval.py`, `eval_llm_perspectives.py` e dalla configurazione di generazione:

- `Qwen_Qwen3.5-9B` (`Qwen/Qwen3.5-9B`);
- `mistralai_Ministral-3-8B-Instruct-2512` (`mistralai/Ministral-3-8B-Instruct-2512`);
- `google_gemma-4-E4B-it` (`google/gemma-4-E4B-it`).

Per ogni dataset/modello il file `no_perspective_*_conversation.csv` viene
riutilizzato per tutte le prospettive, ricalcolando ogni F1 sulle etichette e sul
campione corrente. Il riferimento unico è `aggregation_on_female/male/genx/geny/
genz/boomer/colleague/best_friend`, letto dai JSON `disaggregated_dataset/`.
Non si usano come riferimento le etichette incorporate nei CSV delle inferenze.

Non viene ricostruita o modificata alcuna etichetta. Nel notebook
`updatingSemiaggregationMultiPICoCSC.ipynb` la costruzione usa maggioranza e
pareggio risolto a **1**, nonostante commenti/docstring parlino di sorteggio.
MP usa gli osservatori; CSC include anche lo speaker. CSC ricava la generazione
dall'età con anno di riferimento 2023 quando manca. Queste convenzioni restano
quelle degli artefatti esistenti. L'ammissibilità mantiene il controllo storico
sulla presenza di `aggregation_on_*_n`, oltre alla validità binaria del riferimento.

L'universo di valutazione è quello dei JSON `_final.json` in `../datasets/`,
come nell'analisi precedente. L'abbinamento usa **unique_id come stringa**:
non dipende dall'ordine dei CSV e non converte gli identificativi numerici in float.
Gli ID sono ordinati prima delle statistiche, rendendo anche il ricampionamento
indipendente dall'ordine dei file. I duplicati nei JSON o nei CSV, anche identici,
e gli ID vuoti provocano un errore esplicito; non si sceglie arbitrariamente una riga.

Il campione comune contiene solo ID nel `_final`, con annotazione specifica e
voti disponibili, entrambe le predizioni binarie e prompt compatibili. Si tratta
di una politica di **coppie complete**, coerente con l'esclusione delle predizioni
non valide già presente in `eval.py`: nessun errore viene trasformato in 0 o
eliminato senza registrazione. In caso di dati mancanti si valuta esplicitamente
l'intersezione, si riduce la copertura e si limita l'inferenza a quel campione;
gli errori non osservati possono rendere tale campione selezionato. I confronti
senza coppie sono mantenuti con stato `no_valid_pairs` e metriche mancanti.

Controlli eseguiti sui dati principali e sulla sensibilità:

- zero duplicati, ID vuoti, predizioni non binarie, annotazioni mancanti o
  riferimenti non binari tra gli esempi con voti;
- nessuna coppia mancante o incompatibilità di prompt tra gli esempi ammissibili;
- **copertura appaiata 100%** per tutti i confronti; nessuna perdita dovuta
  all'intersezione delle due condizioni;
- i CSV contengono 1.382 righe MP e 2.946 CSC; il filtro `_final` esclude
  rispettivamente 5 e 41 ID per file, lasciando 1.377 e 2.905 esempi;
- le restanti riduzioni dipendono esclusivamente dall'assenza di voti per la
  prospettiva, come riportato nella tabella seguente (uguale per i tre modelli).

| Dataset | Prospettiva | Esclusi senza voti | N effettivo | Cluster |
|---|---|---:|---:|---:|
| MP | Female | 78 | 1299 | 1218 |
| MP | Male | 42 | 1335 | 1246 |
| MP | Gen X | 257 | 1120 | 1053 |
| MP | Gen Y | 51 | 1326 | 1240 |
| MP | Gen Z | 694 | 683 | 652 |
| MP | Boomer | 1114 | 263 | 263 |
| CSC | Female | 56 | 2849 | 103 |
| CSC | Male | 54 | 2851 | 103 |
| CSC | Gen X | 779 | 2126 | 103 |
| CSC | Gen Y | 96 | 2809 | 103 |
| CSC | Gen Z | 597 | 2308 | 103 |
| CSC | Boomer | 2163 | 742 | 103 |
| CSC | Colleagues | 2049 | 856 | 32 |
| CSC | Best Friends | 856 | 2049 | 71 |

`n_eligible` è il denominatore di `paired_coverage`; `n_samples` il numeratore.
Le colonne `n_invalid_*` riguardano gli ammissibili, mentre
`n_invalid_all_rows_*` contano anche gli output fuori campione.
Prevalenza positiva e F1 sono espressi in scala 0–1. La prevalenza viene
ricalcolata sulle coppie effettive, non sull'intero dataset.

## Comparabilità del protocollo

Per ciascun ID si ricostruisce la conversazione dai JSON originali con lo stesso
procedimento del decoder: si elimina soltanto `u2` dall'ultimo turno e si mantengono
i turni precedenti. La conversazione nei due prompt deve coincidere esattamente
con questa stringa. Il resto del prompt deve corrispondere al template specifico
nel decoder, consentendo soltanto differenze di spaziatura esterne alla
conversazione. Anche le conversazioni dei `_final` sono confrontate con quelle
dei JSON delle etichette. Tutti i controlli passano sui dati valutati.

Questo verifica input `conversation`, task zero-shot, codifica 0/1, assenza della
risposta finale e unica variazione testuale di istruzione relativa alla
prospettiva. I tratti adottano la prospettiva dell'osservatore; le relazioni
aggiungono l'informazione sui due interlocutori, come nei template esistenti.

Il codice attuale documenta `max_new_tokens=32`, `temperature=0` (greedy), batch
32, fino a tre retry degli output invalidi; Qwen/Gemma passano `enable_thinking=False`
con fallback del template, e Ministral usa `no_repeat_ngram_size=8`. La stessa
pipeline è prevista per Neutral e Conditioned entro modello. Il dtype configurato
è float16, mentre il backend Ministral non passa esplicitamente il dtype al loader.

**Limite di provenienza:** i CSV non salvano checkpoint/revisione effettivi,
versioni di librerie/tokenizer/template, parametri di generazione, commit o
manifest dell'esecuzione. L'identità del modello è quindi quella indicata dal
percorso/nome file, non una verifica indipendente del checkpoint. La configurazione
attuale non dimostra che quelle impostazioni siano state usate per tutte le
inferenze salvate, anche perché il decoder riprende file già esistenti.
`protocol_status` esplicita `generation_unverified`. Non sono state trovate
difformità nei prompt osservabili, ma i risultati non isolano con certezza
l'effetto del solo conditioning. Per completare questa verifica servono, per
ciascun file di entrambe le condizioni, log/manifest con modello e revisioni,
configurazione di generazione, chat template, versione del backend e commit.
Non servono nuove predizioni per completare i confronti numerici.

## Statistiche

Metrica principale: **positive-class F1, pos_label=1**, con `zero_division=0`.
Nel task il prompt definisce 1 come "afforded"; i riferimenti rimangono proxy
labels di ironia/sarcasmo. Il delta principale è sempre
**F1 Conditioned − F1 Neutral**, mantenendo un solo vettore di riferimenti.

La valutazione usa le API pubbliche delle librerie già presenti nel progetto:

| Componente | Implementazione |
|---|---|
| F1 Neutral, Conditioned e complementari | `sklearn.metrics.f1_score`, `average="binary"`, `pos_label=1`, `zero_division=0` |
| Intervalli bootstrap | `scipy.stats.bootstrap`, `paired=True`, `method="percentile"`, `confidence_level=0.95` |
| Approximate randomization | `scipy.stats.permutation_test`, `permutation_type="samples"`, `alternative="two-sided"` |
| Correzione Holm | `statsmodels.stats.multitest.multipletests`, `method="holm"` |
| Abbinamento delle predizioni | `pandas.DataFrame.join`, `validate="one_to_one"`, indice `unique_id` |

Resta specifica del progetto la definizione dei cluster, dei filtri, dei prompt
compatibili e del layout delle tabelle. Nelle repliche rimane una piccola funzione
NumPy vettoriale per `2*TP/(2*TP+FP+FN)`, con zero se il denominatore è nullo:
scikit-learn non espone un argomento `axis` per valutare in batch matrici di
conteggi. Richiamare `f1_score` per ogni singola condizione e replica richiederebbe
milioni di chiamate. La formula è verificata contro `f1_score` usando conteggi
pesati TP/FP/FN, inclusi casi degeneri; i punteggi puntuali usano direttamente
scikit-learn. Gli algoritmi di ricampionamento e i test sono delegati a SciPy.

Bootstrap appaiato a cluster con **10.000 repliche**, IC percentile 95%:

1. CSC: cluster `source:key_1`, dove `key_1` è `context_name` nel merger;
   MP: cluster `source:key_2`, dove `key_2` è `post_id` (non il reply ID).
2. Si estraggono con reinserimento tanti cluster quanti quelli osservati e si
   includono tutti gli ID di ogni cluster estratto per entrambe le condizioni.
   N può quindi cambiare tra repliche per cluster di dimensione diversa.
3. Si ricalcolano i due F1 sull'insieme degli esempi estratti, quindi la differenza.
   Il codice somma TP/FP/FN per cluster: è esattamente equivalente al
   ricampionamento degli ID nei cluster, non è una media degli F1 di cluster.

Il motore di ricampionamento è `scipy.stats.bootstrap`, con indici di cluster
come osservazioni e `paired=True`: gli stessi cluster estratti sono quindi usati
per Neutral e Conditioned. `method="percentile"` è esplicito per conservare il
metodo precedente (il default di SciPy sarebbe BCa). La funzione riceve 10.000
repliche e seed riproducibile; si leggono gli estremi da `confidence_interval`,
senza implementare manualmente il ciclo o il calcolo dei quantili. Un solo
cluster produce un intervallo degenere, non informativo sulla variabilità tra
cluster; questo caso è gestito analiticamente. `bootstrap_degenerate` segnala
anche altre distribuzioni di ricampionamento costanti.

**Paired approximate randomization test a cluster**, bilaterale, con **10.000
permutazioni Monte Carlo**, eseguito tramite `scipy.stats.permutation_test`
con `permutation_type="samples"` e `alternative="two-sided"`. Ogni cluster
scambia o mantiene, con probabilità 1/2, l'intero vettore delle due condizioni.
Il riferimento resta fermo. La statistica è il delta F1 complessivo ricalcolato
a ogni replica, non una differenza di medie. SciPy scambia gli indici di interi
vettori TP/FP/FN: i conteggi dello stesso cluster rimangono sempre insieme.

Si adotta la convenzione bilaterale di SciPy, che raddoppia la coda minore:
`p = min(1, 2 * min((1 + count(delta_perm <= delta_obs))/(B+1),
(1 + count(delta_perm >= delta_obs))/(B+1)))`, con `B=10000`.
Gli ex aequo sono inclusi usando la tolleranza relativa di SciPy
(100 volte l'epsilon del tipo floating-point). Il p-value minimo Monte Carlo
è quindi `2/10001`, circa 0,0002; non viene mai dichiarato uguale a zero.
Per campioni piccoli SciPy enumera esattamente tutti i `2^G` scambi se il
numero richiesto li copre: in quel caso il denominatore è `2^G` e non si usa
la correzione +1. `permutation_iterations` riporta il numero effettivo,
`permutation_resamples_requested` quello richiesto e `permutation_exact` la modalità.
Con un solo cluster i due scambi danno delta opposti e il p-value bilaterale
esatto è 1; non esiste potenza per distinguere le condizioni in quel caso.
Nei dati analizzati tutti i confronti usano 10.000 permutazioni Monte Carlo.

Questa implementazione sostituisce il precedente ciclo manuale di scambi.
La convenzione precedente contava `|delta_perm| >= |delta_obs|`: con la
distribuzione esatta simmetrica coincide con il raddoppio della coda minore,
ma una simulazione finita può dare p-value diversi. Anche il generatore degli
scambi di SciPy produce una sequenza diversa a parità di seed iniziale.
F1, delta e IC bootstrap rimangono identici e, sui dati disponibili, non cambia
nessuna decisione dopo Holm. CSV, tabelle e valori di seguito usano SciPy.

L'ipotesi nulla richiede scambiabilità delle condizioni a livello di cluster,
una condizione più forte della sola uguaglianza dell'F1. Il test non usa le
code del bootstrap ordinario e non elimina questo requisito di scambiabilità.
Riferimento API: [SciPy permutation_test](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html).

Il raggruppamento protegge la dipendenza tra esempi che condividono il contesto
originario o il post. Non modella ulteriore dipendenza incrociata dovuta agli
stessi annotatori tra contesti; la generalizzazione rimane al corpus e ai
raggruppamenti documentati. Le proxy labels sono tenute fisse: l'incertezza della
loro costruzione o del campionamento degli annotatori non viene stimata.

Holm a 0,05 conserva le famiglie **dataset × modello** di `eval.py`: sei test per
MP, otto per CSC (tratti e relazioni nella stessa famiglia CSC). Le tabelle
separate non creano nuove famiglie. Eventuali confronti mancanti mantengono
conservativamente il loro posto nella famiglia con p=1 solo per la correzione;
i relativi risultati restano mancanti. Gli **IC 95% sono marginali e non corretti**
per confronti multipli. Un IC che esclude zero non implica significatività dopo
Holm; la decisione corretta usa `p_value_holm`.

Seed base 42. Ogni confronto usa un seed stabile derivato con SHA-256 da
dataset/modello/strategia; bootstrap e permutazione usano flussi distinti,
derivati ulteriormente dal seed del confronto. `comparison_seed` è nel CSV,
il seed base e i conteggi sono anche nel manifest. Non dipende dal `hash()` Python.

## Risultati principali: output/decoders

### Tratti

- **Gemma4-E4B:** su entrambi i dataset migliorano Female, Male, Gen X, Gen Y e
  Gen Z dopo Holm. I delta vanno da +0,0497 a +0,1767 su MP e da +0,0381 a
  +0,1801 su CSC. Boomer peggiora su CSC (−0,0564; p Holm=0,0076); su MP
  il delta −0,0052 resta incerto.
- **Qwen3.5-9B:** su MP migliora Male (+0,0463; p Holm=0,0150), mentre su CSC
  migliorano Female (+0,0338), Male (+0,0611), Gen Y (+0,0595) e Gen Z (+0,0799).
  Boomer peggiora su entrambi: MP −0,1949 (p Holm=0,0060), CSC −0,1471
  (p Holm=0,0016). Gli altri confronti non superano Holm.
- **Ministral3-8B:** nessun confronto dei tratti supera Holm. I delta osservati
  sono piccoli, ma la mancata significatività non dimostra assenza di effetto.

Esempio richiesto, Gen Y:

| Dataset | Modello | N | Neutral F1 | Conditioned F1 | Delta F1 | IC 95% delta | p Holm |
|---|---|---:|---:|---:|---:|---|---:|
| MP | Qwen3.5-9B | 1326 | 0,4432 | 0,4604 | +0,0172 | [−0,0150; +0,0497] | 0,9065 |
| CSC | Qwen3.5-9B | 2809 | 0,4604 | 0,5200 | +0,0595 | [+0,0385; +0,0815] | 0,0016 |
| MP | Gemma4-E4B | 1326 | 0,2904 | 0,4297 | +0,1393 | [+0,1026; +0,1771] | 0,0012 |
| CSC | Gemma4-E4B | 2809 | 0,3593 | 0,4782 | +0,1189 | [+0,0879; +0,1519] | 0,0016 |

### Relazioni (solo CSC)

L'unico miglioramento che supera Holm è **Gemma4-E4B, Best Friends**:
N=2049, 71 cluster, prevalenza positiva 0,3148; Neutral F1=0,3495,
Conditioned F1=0,4451, delta=+0,0957, IC 95% [+0,0647; +0,1277],
p=0,0002, p Holm=0,0016. Colleagues per Gemma ha delta +0,0037 e IC che
comprende zero. Per Qwen i delta sono −0,0212 (Colleagues) e −0,0298
(Best Friends); nessuno supera Holm, anche se l'IC marginale di Best Friends
esclude zero. Nessun confronto delle relazioni di Ministral supera Holm.

Questi risultati misurano un miglioramento rispetto al prompt neutro
**nell'allineamento con le proxy labels della prospettiva**. Non costituiscono
una validazione diretta del giudizio umano sull'affordance di una risposta ironica.
Non implicano neppure che un modello con delta maggiore abbia F1 assoluto maggiore:
le baseline assolute sono differenti. Le conclusioni riguardano soltanto i
dataset, modelli e prospettive sopra specificati, entro i limiti di provenienza.

## Seconda raccolta locale: sensibilità separata

L'inventario ha individuato anche `output/decoders_temp0/` (48 CSV, ignorati da
Git), `output_old/` (modelli precedenti) e `output_old2/` (archivio precedente).
La scelta primaria del percorso canonico deriva dagli script esistenti, non
dai risultati ottenuti. Gli archivi non vengono mescolati o usati come baseline.
Tutti i 48 file `decoders_temp0` differiscono dai corrispondenti canonici; il
nome della cartella non è una prova dei parametri effettivi. Anche qui i prompt
sono compatibili e non ci sono errori di predizione nei dati valutati.

La seconda raccolta viene analizzata con lo stesso protocollo in una directory
separata, senza aggregare p-value o modificare le famiglie della prima:

```powershell
.venv/Scripts/python.exe -X utf8 3_experiments/experiment2/eval_perspectives.py --predictions-dir 3_experiments/experiment2/output/decoders_temp0 --output-dir 3_experiments/experiment2/output/perspectives_temp0
```

Questa sensibilità restituisce lo stesso insieme di risultati significativi
principali, con l'aggiunta di **Female, Qwen, MP** (delta +0,0449;
p Holm=0,0272), che non supera Holm nella raccolta principale. Gemma migliora
ancora nei cinque tratti diversi da Boomer su entrambi i dataset e su Best
Friends; i peggioramenti Boomer di Qwen su MP/CSC e di Gemma su CSC persistono.
Ministral non supera Holm in nessun confronto. Nessuna significatività viene
reinterpretata come conferma di impostazioni di generazione equivalenti tra
raccolte: per questo servono i log indicati sopra.

## Verifiche implementate

Venti test automatici coprono abbinamento con righe permutate, etichette
specifiche fisse anche quando Global differisce, segno del delta/classe positiva,
predizioni identiche (delta e IC esattamente zero, p=1), F1 senza positivi,
duplicati, output invalidi, file/predizioni mancanti, prompt o conversazioni
incompatibili, Holm e famiglie incomplete. Bootstrap e permutazione a cluster
sono confrontati con un'implementazione diretta che estrae gli ID e scambia le
predizioni, verificando che tutti i membri del cluster seguano lo stesso scambio.
Il caso esatto enumera tutti gli scambi; il caso Monte Carlo confronta SciPy
con una ricostruzione diretta delle predizioni per ID. Sono verificati anche
il caso con un solo cluster, il conteggio delle permutazioni per oltre 1023 cluster,
il layout delle tabelle, le parità arrotondate e l'assegnazione degli asterischi
al vincitore effettivo usando esclusivamente i p-value corretti.
La formula vettoriale dell'F1 viene confrontata con scikit-learn su 68 vettori
di conteggi, inclusi tutti i casi elementari e quello senza positivi. Si controlla
che il calcolo puntuale rifiuti valori non binari senza troncarli a interi e che
il join Pandas intercetti duplicati anche quando si salta il caricatore dei CSV.
