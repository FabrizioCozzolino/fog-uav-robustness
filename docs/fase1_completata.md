# Riassunto Fase 1 — Cosa abbiamo ottenuto

*Documento da leggere durante la pausa. Riepiloga il lavoro fatto, i risultati,
le lezioni apprese, e le domande tipiche d'esame su questa fase.*

---

## 0. Lo stato del progetto

```
[✅] Fase 0 — Setup ambiente, scaricamento dataset, struttura progetto
[✅] Fase 1 — Baseline U-Net su VDD (clean)        <-- SEI QUI
[⏳] Fase 2 — GAN per generazione di nebbia (Pix2Pix)
[⏳] Fase 3 — Generazione VDD_foggy
[⏳] Fase 4 — Test robustezza U-Net su VDD_foggy
[⏳] Fase 5 — Re-training U-Net su dati nebbiosi
[⏳] Fase 6 — Report finale + difesa orale
```

Hai completato **circa un terzo del lavoro tecnico**. Le restanti fasi sono più
"derivative" (riusano molto del codice e dei pattern già scritti), quindi non
ti spaventare: la curva di apprendimento più ripida è già alle spalle.

---

## 1. I risultati ufficiali

### Modello v1 — baseline storico

```
Configurazione : U-Net + ResNet-34 pretrained + ImageNet normalization
                 Resize 512x512, no class weights, CrossEntropyLoss standard
                 30 epoch, batch=8, AdamW lr=1e-4, cosine annealing

Risultato      : val mIoU = 0.6524 (epoch 28)
                 Vehicle IoU = 0.0000  <-- problema serio
```

Lo conserviamo nel report come **ablation study** ("ecco cosa succede senza
class weights").

### Modello v2 — modello ufficiale

```
Configurazione : Stesso modello + due cambi chiave:
                 - Resize 768x768 (invece di 512x512)
                 - Class weights inverse_sqrt sulla CE Loss

Risultato val  : mIoU = 0.7602  F1 = 0.8557  acc = 0.9098
Risultato test : mIoU = 0.7168  F1 = 0.8279  acc = 0.8624
```

**Test mIoU = 0.7168** è il numero ufficiale che entra nel report.

### Per-class IoU (val, modello v2)

```
other          0.6525  ##########################
wall           0.5859  #######################
road           0.7374  #############################
vegetation     0.8920  ###################################
vehicle        0.5849  #######################   <-- recuperata da 0.0000!
roof           0.9091  ####################################
water          0.9596  ######################################
```

**Numeri da ricordare per l'esame**: 0.7602 (val) e 0.7168 (test). Se ti chiedono
"quanto fa la tua U-Net su VDD?", rispondi "test mIoU 0.72".

### Confronto baseline vs final

```
                 v1 (baseline)    v2 (final)    Δ
mIoU val         0.6524           0.7602        +10.8 punti
F1  val          0.7328           0.8557        +12.3 punti
vehicle IoU      0.0000           0.5849        +0.585 (RECUPERO COMPLETO)
```

---

## 2. La storia della Fase 1 (da raccontare all'esame)

Una buona presentazione ha sempre una **narrativa**, non solo numeri. Ecco la
nostra in 4 atti:

### Atto 1 — Setup e preparazione (Fase 0)

> Abbiamo configurato l'ambiente Python con PyTorch, scaricato VDD (400 immagini
> aeree con maschere a 7 classi) e Foggy Cityscapes (1500 immagini stradali con
> 3 livelli di nebbia paired). Abbiamo strutturato il progetto in modo
> modulare: dataset, modello, training, evaluation in cartelle separate.

### Atto 2 — Pipeline U-Net + smoke test

> Abbiamo implementato una classe Dataset PyTorch per VDD, una pipeline di
> augmentation con Albumentations (flip, rotate, jitter colori), il modello
> U-Net con encoder ResNet-34 pretrained su ImageNet, e il training loop con
> metriche torchmetrics e logging TensorBoard. Abbiamo verificato la
> correttezza con uno smoke test su 20 immagini in CPU: la loss scendeva e la
> mIoU saliva — la pipeline era pronta.

### Atto 3 — Training reale e diagnostica

> Abbiamo addestrato il primo modello su Colab (GPU T4) per 30 epoch. Il
> risultato era sembrava buono (mIoU 0.6524), ma analizzando le metriche per
> classe abbiamo scoperto che la classe `vehicle` aveva IoU 0.0000 — il
> modello non riconosceva nessun veicolo.

### Atto 4 — Diagnostica e fix

> Abbiamo investigato il problema scrivendo uno script che misura la frequenza
> di ogni classe: i veicoli rappresentavano solo lo 0.6% dei pixel, contro il
> 38% della vegetazione. Inoltre il resize aggressivo 4000x3000 → 512x512
> riduceva l'area dei veicoli di ~46x, facendoli sparire. Abbiamo applicato
> due correzioni standard:
>
> - Aumento della risoluzione a 768x768 (riduzione "solo" 20x)
> - Class weights inverse_sqrt nella CrossEntropyLoss
>
> Il re-training ha portato la mIoU a 0.7602 (+10.8 punti) e il vehicle IoU
> a 0.5849. Validazione finale sul test set: mIoU 0.7168.

**Questa è la storia che vuoi raccontare all'esame.** Mostra:
- Hai costruito una pipeline completa
- Hai misurato i risultati con rigore (val + test, per classe)
- Hai trovato un problema (non l'hai nascosto)
- Hai applicato una soluzione standard e l'hai validata
- Hai documentato tutto

Questo è esattamente come si lavora nella ricerca e nell'industria.

---

## 3. Le 5 lezioni tecniche fondamentali

### Lezione 1 — Le metriche aggregate possono ingannare

**Cosa è successo**: il modello v1 aveva mIoU 0.6524, sembrava OK. Ma una classe
era completamente persa.

**Lezione**: una metrica complessiva è la **media** di metriche per classe.
Una media decente può nascondere fallimenti su classi rare. Per questo nei
problemi multi-classe **dobbiamo sempre guardare le metriche per classe**, non
solo la mIoU complessiva.

**Quando ti accadrà di nuovo**: sempre. Ogni volta che lavori con segmentation o
classification multi-classe, guarda i risultati per classe.

### Lezione 2 — Il class imbalance va sempre verificato

**Cosa è successo**: nel dataset, vegetazione e water erano l'80% dei pixel,
vehicle era lo 0.6%.

**Lezione**: prima di addestrare un modello, **misura le frequenze di classe**
nel training set. È un investimento di 10 secondi che ti risparmia ore di
debugging.

**Soluzioni standard al class imbalance**:
- Class weights nella loss (quello che abbiamo fatto)
- Loss alternativa: Focal Loss, Dice Loss
- Oversampling delle classi rare
- Data augmentation mirato sulle classi rare

### Lezione 3 — La risoluzione conta tantissimo

**Cosa è successo**: a 512x512 i veicoli sparivano dopo il resize. A 768x768
sopravvivevano.

**Lezione**: scegliere la risoluzione di input non è una decisione di
"performance", è una decisione di **cosa il modello può fisicamente vedere**.
Se vuoi rilevare oggetti piccoli, devi avere abbastanza pixel per loro nel
input.

**Trade-off da ricordare**:
- Risoluzione raddoppia → memoria e tempo training x4 (area scala col quadrato)
- A risoluzione più bassa: training veloce, oggetti grandi visibili, oggetti
  piccoli persi
- A risoluzione più alta: training lento, oggetti piccoli visibili, batch più
  piccoli (rischio di gradient noise)

### Lezione 4 — Val e test set servono per cose diverse

**Cosa abbiamo visto**: val mIoU 0.7602 vs test mIoU 0.7168. Il gap è normale.

**Lezione**:
- Il **val set** si usa **durante** il training per scegliere il best
  checkpoint e tarare iperparametri. Le sue metriche hanno un piccolo bias
  ottimistico (selection bias).
- Il **test set** si usa **una sola volta**, alla fine, per misurare la
  performance reale. Le sue metriche sono "oneste".
- **Mai** usare il test set per scegliere il modello, le epoch, gli
  iperparametri. Se lo fai, perdi la garanzia di onestà.

### Lezione 5 — Il transfer learning è gratis

**Cosa è successo**: il nostro encoder ResNet-34 era pretrained su ImageNet
(1M immagini, 1000 classi di oggetti street-level). Eppure ha funzionato
benissimo su immagini aeree (dominio diverso).

**Lezione**: anche con domain gap, le **feature low-level** (bordi, texture,
forme di base) trasferiscono. Iniziare da pretrained è quasi sempre meglio che
iniziare da zero, soprattutto con dataset piccoli (280 immagini di train sono
poche).

**Quando NON fare transfer learning**: solo se il tuo dominio è veramente
esotico (es. immagini mediche con modalità non ottiche, dati satellitari
multispettrali con 13 canali) e devi cambiare l'architettura del primo layer.

---

## 4. Le domande tipiche d'esame su Fase 1

Prova a rispondere ad alta voce, senza guardare. Se non sai rispondere a una,
è quella su cui devi tornare.

**Q1**: Perché U-Net e non DeepLabV3+ o Mask2Former?

> U-Net è il baseline canonico per la segmentation, ben studiato, con
> architettura semplice. Per un primo studio di robustezza, vogliamo un
> modello standard di cui è facile interpretare il comportamento. Inoltre il
> README del progetto la specifica esplicitamente.

**Q2**: Perché 768x768 e non 1024x1024 o 4000x3000 (originale)?

> Compromesso fra accuratezza e tempo di training. A 768x768 i veicoli sono
> ancora visibili (~20x area shrink), ma il batch size resta 4 su una T4.
> A 1024x1024 dovremmo abbassare il batch a 2 e training raddoppia.
> A 4000x3000 nemmeno una immagine starebbe in VRAM.

**Q3**: Perché AdamW invece di SGD?

> AdamW adatta il learning rate per parametro, è più tollerante a hyperparam
> non perfettamente tarati. SGD richiederebbe tuning di lr e momentum. Per un
> progetto di corso, AdamW dà buoni risultati con meno prove.

**Q4**: Cos'è la cosine annealing del learning rate?

> Schedule che riduce il lr lungo una curva coseno da `lr_max` a ~0 lungo
> tutte le epoch. Inizio: lr alto per esplorare, fine: lr basso per rifinire.
> Più stabile e robusta di "step decay" (decadimento a gradini).

**Q5**: Cosa sono i class weights inverse_sqrt? Perché non inverse puro?

> Inverse_sqrt = peso ∝ 1/√freq. Più mite di inverse (∝ 1/freq) che con
> ratio molto estremi (qui ~75x) può destabilizzare il training. Con
> inverse_sqrt il vehicle ha peso ~3-5x le altre classi, abbastanza per
> spostare l'attenzione del modello.

**Q6**: Cosa sono le augmentation che usate? Perché?

> Resize 768x768, flip orizzontale, flip verticale, rotate 90/180/270, jitter
> di luminosità/contrasto, jitter HSV. Le geometriche aiutano perché un drone
> può volare in qualsiasi orientazione (un'immagine aerea è isotropa). Le
> fotometriche simulano variazioni di luce diurna.

**Q7**: Perché il gap val/test (0.76 vs 0.72)?

> Tre motivi: (1) selection bias del val (abbiamo scelto il best checkpoint
> sulla val mIoU), (2) il test ha solo 40 immagini, varianza statistica
> maggiore, (3) split casuale può portare distribuzioni leggermente diverse.
> 4 punti è normale, sotto 1 sarebbe sospetto, sopra 10 sarebbe overfitting.

**Q8**: Cos'è la mIoU? Perché non basta l'accuracy?

> mIoU = media delle Intersection-over-Union per classe. Se predico K classi,
> per ognuna calcolo `|pred ∩ truth| / |pred ∪ truth|` e faccio la media.
> L'accuracy può essere alta solo predicendo la classe più frequente. Una
> classe rara con accuracy alta passa inosservata. mIoU pesa equamente tutte
> le classi.

**Q9**: Cos'è la CrossEntropyLoss? Cosa "vede" il modello?

> Per ogni pixel, il modello produce 7 score (logits), uno per classe. La CE
> calcola `-log(softmax(logits)[true_class])`. Bassa quando la probabilità
> della classe corretta è alta, alta quando è bassa. Il backward propaga
> gradiente che spinge il modello ad aumentare la probabilità della classe
> giusta su ogni pixel.

**Q10**: Hai fatto overfitting?

> No. Alla fine del training v2, train_loss = 0.27 e val_loss = 0.32. Gap
> di solo 0.05. Le augmentation, il weight decay e la selezione del best
> checkpoint sul val hanno fatto il loro lavoro.

---

## 5. Cosa fare adesso (durante la pausa)

In ordine di importanza:

1. **Riposati**. Davvero, una pausa non-strutturata è preziosa quanto le ore di
   lavoro. Il cervello consolida durante il riposo.

2. **Quando hai voglia**, rileggi `docs/guida_studio.md` e `docs/codice_spiegato.md`.
   Adesso che hai visto il training in azione, le pagine sui training loop
   avranno un significato diverso.

3. **Apri TensorBoard in locale** sul tuo PC (puoi farlo anche senza GPU): vai
   nella cartella del progetto e lancia:
   ```
   tensorboard --logdir outputs\runs
   ```
   Aspetta, questo non funziona perché non hai i log in locale. Hai due
   opzioni:
   - Scarica la cartella `tb/` da Drive, mettila in `outputs/runs/<run>/tb/`,
     poi lancia tensorboard
   - Oppure quando rilanci una sessione Colab, apri TensorBoard direttamente
     lì come abbiamo fatto

4. **Visita il repo GitHub** (https://github.com/FabrizioCozzolino/fog-uav-robustness)
   e guardalo dal browser. Vedere i tuoi file pubblicati, ben strutturati, è
   gratificante e ti dà un colpo d'occhio del progetto.

5. **Prova a spiegare il progetto** a una persona non tecnica. Anche un
   familiare. Se riesci a fargli capire "cosa fa" e "perché", l'hai capito tu.

---

## 6. Cosa ti aspetta nella Fase 2 (preview)

Quando torni, attaccheremo il **Pix2Pix**. È un GAN (Generative Adversarial
Network) che impara a trasformare un'immagine in un'altra. Nel nostro caso:
**immagine clean → immagine nebbiosa**.

Il setting è interessante:
- **Generator**: una rete (U-Net-like!) che produce immagini nebbiose
- **Discriminator**: una seconda rete che prova a distinguere "nebbia vera"
  da "nebbia generata"
- **Adversarial loss**: il generator vince se inganna il discriminator, il
  discriminator vince se non viene ingannato

Il training è **competitivo** e più delicato di una U-Net standard:
- Loss più rumorose (osciliano molto)
- Iperparametri più sensibili
- Tempo di addestramento simile (~30-45 min)

Ma molte cose le hai già viste:
- Stessi tool (PyTorch, Albumentations, TensorBoard)
- Stessa pipeline di lavoro (locale debug → Colab training)
- Stesso pattern dataset (Foggy Cityscapes è già paired)

Quando tornerai, ti spiegherò passo per passo. Per ora, **basta riposare**.

---

*Documento aggiornato il 28 aprile 2026, dopo il completamento della Fase 1.*

**Risultati ufficiali**: U-Net val mIoU 0.7602 / test mIoU 0.7168 su VDD clean.

Buona pausa.
