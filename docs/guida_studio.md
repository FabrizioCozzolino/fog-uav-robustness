# Guida allo studio del progetto

**Per**: ripasso personale ed esame orale
**Argomento**: semantic segmentation su VDD con U-Net, con focus su robustezza sotto nebbia GAN-generata
**Stato attuale**: **PROGETTO COMPLETATO** (Fasi 0-5 completate, mancano solo report scritto e difesa orale)

> Documento aggiornato il 9 maggio 2026, dopo i risultati definitivi della Fase 5
> (recupero della robustezza al ~82-83% via augmentazione foggy).

---

## 0. Il quadro generale (da raccontare all'esame in 60 secondi)

> *"Il progetto consiste nel valutare la robustezza di un modello di semantic segmentation
> U-Net applicato a immagini aeree da drone (dataset VDD), quando queste immagini
> vengono degradate da nebbia sintetica generata da un GAN. La pipeline è divisa in
> cinque fasi: (1) addestramento della U-Net su VDD clean, (2) addestramento di un
> GAN di tipo Pix2Pix che impara a trasformare immagini clean in immagini nebbiose
> a partire dal dataset Foggy Cityscapes, (3) applicazione del GAN a VDD per ottenere
> VDD_foggy a due livelli di severità (medium e dense), (4) test della U-Net originale
> su VDD_foggy per misurare il calo di performance, (5) re-training della U-Net su
> un mix di dati clean + foggy per verificare il recupero di robustezza. Il risultato
> finale: il modello aumentato (v3) recupera circa l'82% della perdita su nebbia media
> e l'83% su nebbia densa, senza degradare le prestazioni sul dato clean."*

Questo è il "paragrafo di apertura" che ti serve saper ripetere a memoria.

---

## 1. I numeri ufficiali da memorizzare

### I 6 risultati principali del progetto

| Modello | Test set | mIoU | Note |
|---------|----------|------|------|
| U-Net **v1** (baseline 512×512, no class weights) | clean (val) | 0.6524 | ablation, vehicle IoU = 0 |
| **U-Net v2** (768×768, inverse_sqrt weights) | **clean (test)** | **0.7168** | baseline ufficiale |
| U-Net v2 | foggy medium (test) | 0.6652 | -7.2% calo |
| U-Net v2 | foggy dense (test)  | 0.5377 | -25.0% calo |
| **U-Net v3** (mixed: clean + medium + dense) | **clean (test)**  | **0.7264** | +0.97% (regolarizzazione) |
| **U-Net v3** | **foggy medium (test)** | **0.7076** | recupero **82%** |
| **U-Net v3** | **foggy dense (test)**  | **0.6866** | recupero **83%** |

### Pix2Pix (Fase 2)
- Generator + Discriminator, training 50 epoch su Colab T4
- Best **val L1**: 0.0385 (medium fog), 0.0468 (dense fog)
- Best epoch: 38 per entrambi (curiosa coincidenza)

### Recupero della robustezza (Fase 5 vs Fase 4)
- Recovery medium = (0.7076 - 0.6652) / (0.7168 - 0.6652) = **82%**
- Recovery dense = (0.6866 - 0.5377) / (0.7168 - 0.5377) = **83%**

---

## 2. Cos'è la semantic segmentation

### Concetto

La **semantic segmentation** assegna a **ogni pixel** di un'immagine un'etichetta di classe.
Non dice solo "c'è una macchina" (image classification) né "c'è una macchina nel rettangolo
X" (object detection), ma disegna esattamente il contorno pixel-per-pixel delle macchine,
delle strade, degli edifici, ecc.

Input:  immagine RGB di shape `(H, W, 3)`, valori 0-255
Output: mappa di etichette `(H, W)` dove ogni pixel è un intero `0..N-1` (N = numero classi)

Nel nostro caso N=7: `0=other, 1=wall, 2=road, 3=vegetation, 4=vehicle, 5=roof, 6=water`.

### Perché è difficile

- I bordi degli oggetti sono ambigui (dove finisce il tetto e comincia il muro?)
- Le dimensioni degli oggetti variano enormemente (un veicolo da 50 pixel vs una strada da 500 000 pixel)
- Serve sia contesto globale ("siamo in una scena urbana") sia dettaglio locale (bordi precisi)

### Cosa vuol dire "aerea da drone"

Le immagini non sono prese dal livello stradale, ma dall'alto (30°, 60°, 90° dal nadir).
Questo cambia completamente l'aspetto degli oggetti: un'auto vista dall'alto è un rettangolino,
non un profilo laterale.

---

## 3. I dataset

### VDD (Varied Drone Dataset)

- 400 immagini aeree reali, 4000×3000 pixel, 7 classi
- Split già fatto dagli autori: 280 train / 80 val / 40 test (70/20/10)
- Formato: immagini `.JPG` RGB, maschere `.png` grayscale dove il valore del pixel **è** il class ID

### Foggy Cityscapes

- 500 immagini **stradali** (street-level) ripetute in 3 versioni:
  - `No_Fog`: immagine originale
  - `Medium_Fog`: nebbia media (visibilità ~300m)
  - `Dense_Fog`: nebbia densa (visibilità ~150m)
- È **paired**: lo stesso file `001.png` esiste in tutte e tre le cartelle ed è la stessa scena
- La nebbia è **sintetica**, generata col modello fisico di Koschmieder (scattering atmosferico)
  applicato ai depth map di Cityscapes

### Perché due dataset?

VDD non ha versioni nebbiose delle sue immagini. Strategia: *"imparo a mettere
nebbia in modo realistico da un altro dataset che ha coppie clean/foggy, e poi applico
quella nebbia a VDD"*. Questo è **domain transfer** e il GAN è il motore che lo realizza.

### Domanda d'esame

- *"Perché usate un dataset street-level per imparare la nebbia e poi la applicate ad
  immagini aeree?"* → Non esistono dataset aerei nebbiosi pubblici con coppie clean/foggy.
  È un limite metodologico dichiarato. Una parte del progetto è proprio verificare se la
  nebbia impara su scene stradali generalizzi a scene aeree (e i risultati dicono che sì,
  generalizza abbastanza bene).

---

## 4. PyTorch e i tensori

### Cos'è PyTorch

Una libreria Python per deep learning. I suoi oggetti fondamentali sono:

- **Tensor**: array multidimensionale di numeri (può stare in GPU e sa calcolare i gradienti)
- **Module**: un pezzo di rete neurale
- **Dataset**: classe che rappresenta una collezione di campioni
- **DataLoader**: carica campioni a batch dal Dataset, in modo parallelizzato

### Cos'è un tensor

Un **tensor** è un array multidimensionale di numeri. Si distinguono per **rank** (numero di dimensioni):

| Dimensioni | Nome | Esempio | Shape |
|-----------|------|---------|-------|
| 0 | Scalare | `3.14` | `()` |
| 1 | Vettore | `[1, 2, 3]` | `(3,)` |
| 2 | Matrice | tabella di numeri | `(H, W)` |
| 3 | Tensor 3D | un'immagine RGB | `(3, 224, 224)` |
| 4 | Tensor 4D | batch di immagini | `(8, 3, 224, 224)` |

Tre attributi principali:
- `shape`: le dimensioni
- `dtype`: il tipo dei numeri (`float32`, `int64`, ecc.)
- `device`: dove vive (`cpu`, `cuda`)

Due "superpoteri" rispetto a numpy:
- **Autograd**: ogni operazione viene "registrata" e PyTorch può calcolare automaticamente
  i gradienti (chain rule). Permette `loss.backward()`.
- **GPU**: se `tensor.to('cuda')`, le operazioni girano su GPU NVIDIA, 30-100x più veloci.

### Operazioni che confondono all'inizio

```python
x.shape                 # vedi dimensioni
x.permute(2, 0, 1)      # riordina dimensioni: HWC -> CHW
x.view(3, 4)            # cambia forma
x.unsqueeze(0)          # aggiunge dim di size 1
x.argmax(dim=1)         # indice del massimo lungo una dimensione
```

L'operazione **`argmax(dim=1)`** è quella che converte i logits del modello (shape
`(B, 7, H, W)`) nelle predizioni finali (shape `(B, H, W)`): per ogni pixel sceglie la
classe con punteggio massimo.

### Il protocollo Dataset

```python
class MioDataset(Dataset):
    def __init__(self, ...):  # leggo paths, NON i file ancora
    def __len__(self):  # quanti elementi ho
    def __getitem__(self, idx):  # restituisce il campione idx
```

Il Dataset apre i file solo quando viene richiesto un campione (lazy loading), così non
tiene tutto in RAM.

### Detail importante: BGR → RGB

OpenCV per motivi storici carica le immagini in ordine BGR (Blue-Green-Red). Se non
convertiamo, il modello pretrained (che si aspetta RGB) dà predizioni pessime.

---

## 5. Le augmentations (`src/utils/transforms.py`)

### Cosa sono

Trasformazioni casuali applicate alle immagini di training per aumentare artificialmente
la varietà dei dati. Aiutano contro l'**overfitting**.

### Le nostre scelte per il train

```python
A.Resize(image_size, image_size)        # 768×768 (v2/v3)
A.HorizontalFlip(p=0.5)
A.VerticalFlip(p=0.5)
A.RandomRotate90(p=0.5)
A.RandomBrightnessContrast(...)
A.HueSaturationValue(...)
A.Normalize(mean=IMAGENET, std=IMAGENET)
ToTensorV2()
```

### Perché queste e non altre

- **Flip e RandomRotate90**: un'immagine aerea è isotropa. Un drone può essere orientato in
  qualsiasi direzione, ruotare di 90° produce una vista perfettamente valida.
- **Jitter di colori**: simula variazioni di luce diurna.
- **Niente scale/zoom**: il resize standardizzato basta.

### Train vs val transforms

- **Train**: include augmentation casuali
- **Val/Test**: solo Resize + Normalize + ToTensor — **zero casualità**

La validazione deve essere **deterministica**.

---

## 6. L'architettura U-Net

### Idea generale

U-Net è una rete a forma di "U" (Ronneberger 2015). Due metà:

- **Encoder** (discendente): riduce risoluzione, aumenta canali. Cattura il **contesto globale**.
- **Decoder** (ascendente): risale alla risoluzione originale tramite upsampling.
- **Skip connections**: collegamenti diretti encoder→decoder per recuperare i dettagli spaziali.

Output finale: tensor `(B, N_classi, H, W)`. Con `argmax(dim=1)` ottieni la mappa di classi.

### Encoder pretrained: ResNet-34

Invece di scrivere un encoder da zero, usiamo **ResNet-34** pretrained su ImageNet.
Vantaggi:
- Inizializzazione intelligente
- Convergenza veloce (poche epoch)
- Generalizza meglio con dataset piccoli (280 train images)

```python
model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", classes=7)
```

Numero di parametri totali: **24.4M**.

---

## 7. Il training loop (Fase 1)

### Schema generale

Per ogni epoch:

```
TRAIN:
    model.train()
    for ogni batch (images, masks):
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

VAL:
    model.eval()
    with torch.no_grad():
        for ogni batch:
            logits = model(images)
            metriche.update(logits, masks)
    if val_mIoU > best_mIoU:
        torch.save(...)
```

### CrossEntropyLoss

```
loss(logits, target) = -log(softmax(logits)[target])
```

In italiano: "quanto bassa è la probabilità assegnata dal modello alla classe corretta?".

### AdamW + Cosine Annealing

- AdamW: variante di Adam con weight decay corretto
- Cosine Annealing: il lr scende lungo una curva coseno da `lr_max` a ~0

### `model.train()` vs `model.eval()`

Alcuni layer (Dropout, BatchNorm) si comportano diversamente:
- Dropout: attivo in train, inattivo in eval
- BatchNorm: in train usa stats del batch corrente; in eval usa stats medie accumulate

Dimenticare `model.eval()` in validazione → metriche scorrette. **Errore classico.**

---

## 8. Le metriche

### mIoU (mean Intersection over Union)

Per classe singola:
```
IoU = | predizione ∩ verità | / | predizione ∪ verità |
```

**mIoU** = media degli IoU di tutte le classi. Pesa equamente classi rare e frequenti.

### Perché mIoU e non accuracy

L'accuracy può essere alta predicendo sempre la classe più frequente. mIoU pesa equamente
tutte le classi, quindi non si lascia ingannare. È il motivo per cui il bug "vehicle IoU = 0"
dell'U-Net v1 era così grave: nascosto dall'accuracy alta.

---

## 9. Il problema vehicle e come l'abbiamo risolto (Fase 1)

Questa è la **prima storia da raccontare all'esame**. Mostra rigore scientifico.

### Atto 1: training v1

Configurazione: 512×512, no class weights, CrossEntropyLoss standard, 30 epoch.
Risultato: **val mIoU = 0.6524** all'epoch 28. Sembra OK.

### Atto 2: la diagnostica per classe

```
other          0.6217
wall           0.5214
road           0.6985
vegetation     0.8839
vehicle        0.0000   ← ALERT
roof           0.8899
water          0.9511
```

**Il modello non riconosce nessun veicolo.**

### Atto 3: l'investigazione

```
[TRAIN] 280 immagini totali
  Immagini con vehicle  : 170 (60.7%)
  Pixel vehicle totali  : 21,041,510
  Frazione di pixel    : 0.6262%
```

I veicoli sono **presenti** ma occupano solo lo 0.6% dei pixel. Doppia causa:
1. **Class imbalance**: la CE "premia" il modello che ignora la classe rara
2. **Aliasing dal resize**: a 512×512, un'auto da 50×30 → 6×4 pixel = sparisce

### Atto 4: il fix (v2)

Due interventi sinergici:

**A) Class weights inverse_sqrt nella CE**

```python
weights = 1.0 / np.sqrt(freqs)
weights = weights / weights.mean()
criterion = nn.CrossEntropyLoss(weight=weight_tensor)
```

`inverse_sqrt` invece di `inverse` puro: `1/freq` darebbe pesi ~150x per vehicle, troppo
aggressivo, destabilizza il training. `1/√freq` dà ~3-5x.

**B) Risoluzione 768×768 invece di 512×512**

Riduzione di area "solo" 20× invece di 46×. I veicoli restano più grandi.

### Atto 5: i risultati

| Classe | v1 | v2 | Δ |
|--------|----|----|----|
| **vehicle** | **0.0000** | **0.5849** | **+0.585** |
| **mIoU** | **0.6524** | **0.7602** | **+0.108** |

**+58.5 punti su vehicle, +10.8 punti complessivi**.

---

## 10. Val set vs Test set

### Cosa fa il val set

Durante il training, dopo ogni epoch, validiamo. Usiamo le metriche per:
- Scegliere il **best checkpoint**
- Decidere quando smettere il training
- Tarare iperparametri

### Cosa fa il test set

Usato **una sola volta** alla fine. Misura la performance reale.

### Perché serve la distinzione

**Selection bias**: se sceglo il best sul val, le metriche su val sono leggermente
ottimistiche. Il test è "vergine".

### Il gap normale (nostri numeri)

- val mIoU = 0.7602
- test mIoU = 0.7168
- Gap = ~4 punti (normale)

> Mai usare il test set per scegliere il modello. Se lo fai, perdi la garanzia di onestà.

---

## 11. Il GAN: Pix2Pix (Fase 2)

### Cos'è un GAN

GAN = Generative Adversarial Network. **Due reti** che si sfidano:

- **Generator (G)**: produce immagini fake. Vuole **ingannare** il discriminator.
- **Discriminator (D)**: dice "real o fake?". Vuole **smascherare** le fake.

Training competitivo: l'equilibrio sano è D incerto (~50%), perché G inganna efficacemente.

### Perché Pix2Pix e non CycleGAN

- **Pix2Pix**: richiede dati paired. Più semplice, training più stabile, risultati più accurati.
- **CycleGAN**: lavora con dati unpaired. Più complesso.

Noi abbiamo dati paired (le 500 triplette di Foggy Cityscapes), quindi Pix2Pix.

### Architettura

- **Generator**: U-Net con ResNet-34 encoder pretrained, output 3 canali RGB, attivazione `tanh` finale (output in [-1, +1])
- **Discriminator**: 70×70 PatchGAN, ~2.77M parametri (9× più piccolo del G), output 30×30 score map

### Loss

- **Discriminator**: `BCE(D(real), 1) + BCE(D(fake), 0)` — vuole distinguere
- **Generator**: `BCE(D(fake), 1) + 100 × L1(fake, real)` — vuole ingannare D **e** essere fedele al target

Il `100` (`λ_L1`) è il fattore di Pix2Pix paper: garantisce fedeltà pixel-per-pixel.

### Trick tecnici

1. **Input al D = concat(clean, foggy)** → conditional GAN, D giudica la coerenza
2. **`tanh` output e mean=0.5/std=0.5 normalize** → range coerente [-1, +1]
3. **`additional_targets={'foggy': 'image'}` Albumentations** → augmentation paired
4. **DCGAN-style init** (`N(0, 0.02²)`) → stabilità a inizio training
5. **Adam betas=(0.5, 0.999)** → standard per GAN, più stabile di Adam default

### Risultati (Fase 2)

| Modello | Best epoch | Best val L1 | Tempo |
|---------|-----------|-------------|-------|
| medium fog | 38 | 0.0385 | 57 min |
| dense fog | 38 | 0.0468 | 57 min |

### Dinamica adversariale osservata

D ha "vinto" alle ultime epoch (D(real)→1, D(fake)→0). Ma:
- `--save best` salva il checkpoint del miglior epoch (38), non dell'ultima
- L1 ha continuato a scendere indipendentemente (peso 100× maschera l'instabilità adversariale)

Questa è la **classica dinamica oscillante dei GAN**.

---

## 12. Generazione VDD foggy (Fase 3)

### Il problema

Vogliamo applicare i 2 Pix2Pix addestrati a Foggy Cityscapes (256×256) alle immagini di
VDD (4000×3000, dominio aereo).

### Multi-scala

Abbiamo generato **4 dataset**:
- `VDD_foggy_medium_256`: applicato a 256, salvato a 768
- `VDD_foggy_medium_768`: applicato a 768, salvato a 768
- `VDD_foggy_dense_256`: applicato a 256, salvato a 768
- `VDD_foggy_dense_768`: applicato a 768, salvato a 768

### Bug delle maschere (lezione tecnica importante)

**Errore iniziale**: avevo copiato le maschere come erano (4000×3000) ma le immagini foggy
erano 768×768. Albumentations richiede shape uguali → crash.

**Fix**: ridimensionare anche le maschere a 768×768 con `cv2.INTER_NEAREST` (nearest-neighbor),
**non** bilineare. Le maschere contengono class ID discreti (0-6), bilineare produrrebbe
valori intermedi (es. 2.5) invalidi.

### Domain gap osservato

La nebbia di Foggy Cityscapes è **dipendente dalla profondità** (Koschmieder model). Su
immagini stradali si vede chiaramente: vicino nitido, lontano biancastro.

Su immagini aeree il concetto di "lontano" è diverso: in viste nadir, tutti i pixel sono
a distanza simile dal sensore. Il GAN applica un effetto **abbastanza uniforme**, che è
meno realistico per scene oblique. **Limitazione metodologica, da dichiarare nel report.**

---

## 13. Robustezza (Fase 4)

### Risultati sulla U-Net v2 (allenata su clean)

| Test set | mIoU | Calo vs clean |
|----------|------|--------------|
| clean | 0.7168 | (baseline) |
| medium @256 | 0.2601 | **-63.7%** ⚠️ |
| medium @768 | 0.6652 | -7.2% |
| dense @256 | 0.2086 | **-70.9%** ⚠️ |
| dense @768 | 0.5377 | -25.0% |

### Lezione importante: @256 invalidato

Il calo enorme di @256 NON è dovuto alla nebbia ma agli **artefatti di upscaling**: il
Pix2Pix ha generato immagini 256×256 e poi abbiamo upsamplato a 768×768. La perdita di
risoluzione domina l'effetto della nebbia.

**Per il report: usiamo solo @768.** Il @256 viene riportato come "ablation che mostra
perché non è una scelta valida".

### I numeri ufficiali

| Test set | mIoU | Calo |
|----------|------|------|
| clean | 0.7168 | - |
| **foggy medium @768** | **0.6652** | **-7.2%** |
| **foggy dense @768** | **0.5377** | **-25.0%** |

### Per classe (dense @768)

Le classi che soffrono di più sotto nebbia densa: `roof` (-36%), `other` (-38%),
`road` (-31%), `water` (-28%). Le più resilienti: `vegetation` (-12.5%), `vehicle` (-14%),
`wall` (-15%).

**Pattern**: superfici grandi/uniformi soffrono di più. Texture distinte (vegetazione,
veicoli) tengono meglio.

---

## 14. Recupero della robustezza (Fase 5)

### Strategia

Re-training della U-Net su un mix bilanciato:
- 280 immagini clean (VDD)
- 280 immagini foggy_medium_768
- 280 immagini foggy_dense_768
= **840 immagini totali**

PyTorch `ConcatDataset` unisce i tre dataset, il DataLoader con `shuffle=True` li mescola
ad ogni epoch.

Stesse hyperparameters della v2 (768×768, batch 4, AdamW 1e-4, cosine annealing, class
weights inverse_sqrt) per **isolare l'effetto della data augmentation**.

### Risultati definitivi

| Test set | v2 (clean only) | v3 (mixed) | Δ |
|----------|----------------|-----------|------|
| clean | 0.7168 | **0.7264** | **+0.0097** 🚀 |
| foggy medium | 0.6652 | **0.7076** | **+0.0424** |
| foggy dense | 0.5377 | **0.6866** | **+0.1490** |

### Recovery rate

- Medium: (0.7076 - 0.6652) / (0.7168 - 0.6652) = **82%**
- Dense: (0.6866 - 0.5377) / (0.7168 - 0.5377) = **83%**

### Cosa è notevole

1. **Clean migliora** (+0.97%) — l'augmentazione foggy ha funzionato anche come
   regolarizzatore, non solo come specializzazione
2. **Recupero ~82-83%** in entrambi i casi (dense e medium) — coerenza che dimostra che
   il fenomeno è solido, non casuale
3. **Win-win**: nessun trade-off, miglioramento ovunque

### Per classe (dense @768)

| Classe | v2 dense | v3 dense | Recovery |
|--------|----------|----------|----------|
| **roof** | 0.5249 | **0.8038** | massimo |
| **water** | 0.6501 | **0.9148** | massimo |
| **other** | 0.3568 | 0.5348 | forte |
| **road** | 0.4650 | 0.6067 | forte |
| **vegetation** | 0.7647 | 0.8596 | moderato |
| **wall** | 0.5233 | 0.5873 | lieve |
| **vehicle** | 0.4789 | 0.4993 | minimo |

Le classi che avevano sofferto di più nella Fase 4 (roof, water) sono quelle che recuperano
di più: la v3 "aggiusta" i punti deboli del modello v2.

---

## 15. La narrativa completa per l'esame (la storia in 5 atti)

### Atto 1 — Setup e baseline

> Abbiamo configurato l'ambiente PyTorch su Colab, scaricato VDD (400 immagini aeree con
> 7 classi) e Foggy Cityscapes (1500 immagini con 3 livelli di nebbia paired). Abbiamo
> implementato una U-Net con encoder ResNet-34 pretrained ImageNet e ottenuto un primo
> baseline su VDD clean.

### Atto 2 — Diagnostica e fix vehicle

> Il primo training ha dato val mIoU 0.6524, ma la classe `vehicle` aveva IoU = 0.0000.
> Abbiamo investigato: i veicoli occupano solo lo 0.6% dei pixel, e il resize 4000×3000 →
> 512×512 li faceva sparire. Abbiamo applicato due fix: risoluzione 768×768 e class
> weights inverse_sqrt nella CrossEntropyLoss. Il modello v2 ha raggiunto val mIoU 0.7602
> (+10.8 punti) e vehicle IoU 0.5849. Test mIoU ufficiale: 0.7168.

### Atto 3 — GAN per nebbia

> Abbiamo addestrato due Pix2Pix (medium e dense) su Foggy Cityscapes paired. Generator
> = U-Net, Discriminator = 70×70 PatchGAN, loss = BCE adversarial + 100×L1. Best val L1:
> 0.0385 (medium), 0.0468 (dense), all'epoch 38 per entrambi. La dinamica adversariale ha
> oscillato, con D che vinceva alle ultime epoch — ma il --save-best ha catturato il miglior
> equilibrio.

### Atto 4 — Misura del calo

> Abbiamo applicato i due GAN a tutto VDD generando 4 dataset (medium/dense × 256/768).
> Testando la U-Net v2 su questi: medium @768 = 0.6652 (-7.2%), dense @768 = 0.5377
> (-25.0%). Le varianti @256 hanno mostrato un calo molto più grave (-64% a -71%) ma
> non per la nebbia: per gli artefatti di upscaling. Le abbiamo riportate come ablation
> che giustifica la scelta di lavorare a 768×768.

### Atto 5 — Recupero via augmentazione

> Abbiamo riallenato la U-Net su un mix bilanciato di clean + foggy_medium + foggy_dense
> (840 immagini totali, stesse hyperparameters). Risultati: clean 0.7264 (+0.97%), medium
> 0.7076, dense 0.6866. Recupero dell'82% e 83% delle perdite. La conclusione: il
> data augmentation con nebbia GAN-generata è efficace per rendere robusta la U-Net,
> anche con domain gap (Cityscapes → drone).

---

## 16. Le 25 domande tipiche d'esame

**Q1: Perché U-Net e non DeepLabV3+ o Mask2Former?**
> Baseline canonico, ben studiato, semplice da training. Il README del progetto la
> specifica esplicitamente.

**Q2: Perché 768×768 e non 512×512 o 1024×1024?**
> 512 era troppo aggressivo per oggetti piccoli (vehicle spariva a 6×4 pixel). 1024
> raddoppierebbe il tempo di training. 768 è un compromesso che mantiene visibili i
> veicoli con costo contenuto.

**Q3: Perché AdamW invece di SGD?**
> Adatta lr per parametro, più tollerante a hyperparam non perfetti. Per progetti di
> corso dà buoni risultati con meno tuning.

**Q4: Cos'è la cosine annealing del learning rate?**
> Schedule che riduce il lr lungo curva coseno da `lr_max` a ~0. Inizio: lr alto per
> esplorare, fine: lr basso per rifinire.

**Q5: Come avete gestito il class imbalance?**
> Diagnosticato: vehicle = 0.0000 dopo il primo training. Misurato le frequenze: 0.6%
> per vehicle vs 38% vegetation. Applicato due fix: risoluzione 768 e CE con pesi
> inverse_sqrt. Vehicle IoU passa da 0 a 0.585.

**Q6: Perché inverse_sqrt e non inverse puro?**
> Inverse puro darebbe pesi ~150× alla classe vehicle, destabilizza il training.
> Inverse_sqrt dà ~3-5×, abbastanza per spostare l'attenzione senza creare instabilità.

**Q7: Cos'è la mIoU? Perché non basta l'accuracy?**
> mIoU = media degli IoU per classe. Accuracy può essere alta predicendo solo la classe
> più frequente; mIoU pesa equamente tutte le classi.

**Q8: Cos'è la CrossEntropyLoss?**
> Per ogni pixel, il modello produce 7 score (logits). CE calcola
> `-log(softmax(logits)[true_class])`. Bassa quando il modello è sicuro della classe
> giusta.

**Q9: Hai fatto overfitting?**
> No. Alla fine train_loss = 0.27, val_loss = 0.32. Gap di 0.05. Le augmentation, weight
> decay e selection del best checkpoint hanno fatto il loro lavoro.

**Q10: Differenza val vs test set?**
> Val si usa durante il training per scegliere il best checkpoint. Bias ottimistico
> (selection bias). Test si usa una sola volta, alla fine, dà metriche oneste.

**Q11: Cosa contiene un file `.pth`?**
> Dict Python con `model_state_dict`, `optimizer_state_dict`, epoch, mIoU, args. Si
> carica con `torch.load()` e si applica con `load_state_dict()`.

**Q12: Cos'è un GAN?**
> Generative Adversarial Network. Due reti: Generator (produce fake) e Discriminator
> (distingue real da fake). Training competitivo; l'equilibrio sano è D al 50%.

**Q13: Perché Pix2Pix e non CycleGAN?**
> Pix2Pix richiede dati paired (più semplice, più stabile). Foggy Cityscapes è già
> paired (3 livelli di nebbia per stessa scena), quindi Pix2Pix è la scelta naturale.

**Q14: Perché PatchGAN come discriminator?**
> Giudica patch 70×70 invece dell'intera immagine. Più stabile, focalizza su texture
> locali (dove la nebbia "vive"), generalizza meglio. È la scelta del paper Pix2Pix.

**Q15: Cos'è la lambda L1 = 100?**
> Peso del termine L1 nella loss del Generator. Senza, G produrrebbe immagini "creative"
> ma scollegate dal target. Con solo L1 (no GAN), produrrebbe immagini sfuocate.
> Insieme: realistiche **e** fedeli. Il 100 è il default del paper.

**Q16: Perché output del Generator in [-1, +1]?**
> Il Generator finisce con `tanh`, che mappa in [-1, +1]. Per coerenza, anche i target
> sono normalizzati (mean=0.5, std=0.5) → [-1, +1]. La L1 confronta cose nello stesso
> range.

**Q17: Cosa indica D(real) → 1 e D(fake) → 0?**
> Significa che il Discriminator ha "vinto": distingue perfettamente real da fake. È
> un segnale di squilibrio del GAN, ma se la L1 sta scendendo (peso 100×), il Generator
> continua a imparare.

**Q18: Perché 50 epoch per Pix2Pix e 30 per U-Net?**
> Tre motivi: (1) il Pix2Pix ha decoder e Discriminator partiti da zero, mentre la U-Net
> ha solo l'encoder pretrained; (2) image-to-image translation è più difficile della
> segmentation per dinamica adversariale; (3) il paper Pix2Pix raccomanda 100-200 epoch,
> 50 è un compromesso pragmatico. La U-Net a 30 era già a plateau.

**Q19: Perché applicare il GAN a 768 e non a 256?**
> Il @256 introduce artefatti di upscaling che dominano l'effetto della nebbia (mIoU
> scende a 0.26). A 768 il GAN è applicato direttamente alla risoluzione di evaluation
> della U-Net, niente artefatti.

**Q20: Domain gap del nostro setup?**
> Pix2Pix è stato addestrato su immagini stradali (Cityscapes), applicato a immagini
> aeree (VDD). La nebbia di Foggy Cityscapes è dipendente dalla profondità (Koschmieder),
> ma in viste aeree quasi-nadir non c'è un vero gradiente di profondità. Il GAN applica
> un effetto pseudo-uniforme, che è meno realistico ma sufficiente per misurare il
> domain gap.

**Q21: Perché un mix di clean + medium + dense per il re-training?**
> Per coprire tutto lo spettro di severità senza sacrificare troppo il dato originale.
> Mantiene il clean in training previene il "catastrophic forgetting" del dominio
> sorgente.

**Q22: Perché 840 immagini e non 1120 (con anche @256)?**
> Le @256 sono contaminate da artefatti di upscaling. Includendole nel training, il modello
> imparerebbe a essere robusto a "immagini sfuocate", non a nebbia. Limita la robustezza
> al fenomeno reale.

**Q23: Perché lo stesso encoder ImageNet anche per Pix2Pix?**
> Riusiamo l'expertise: il Generator deve "capire" l'immagine di input per applicarvi la
> nebbia. ResNet-34 pretrained gli dà una buona base per estrarre feature. Solo il decoder
> del G parte da zero.

**Q24: Cosa cambia con `--data-roots` invece di `--data-root`?**
> `--data-root` accetta un singolo path → training set è il train split di quel path.
> `--data-roots` accetta una lista → ConcatDataset di PyTorch unisce i train splits di
> tutti i path. Permette training su mix di dataset senza scrivere codice nuovo.

**Q25: Quali sono i numeri ufficiali del progetto?**
> Test mIoU: U-Net v2 clean 0.7168, foggy medium 0.6652 (-7.2%), foggy dense 0.5377
> (-25.0%). U-Net v3 (mixed) clean 0.7264, foggy medium 0.7076, foggy dense 0.6866.
> Recupero dell'82% sul medium, 83% sul dense.

---

## 17. Glossario

| Termine | Definizione breve |
|---------|-------------------|
| Tensor | Array multidimensionale di numeri (può stare in GPU) |
| Backbone / Encoder | Parte della rete che estrae feature dalle immagini |
| Logits | Output grezzo del modello, prima del softmax |
| Softmax | Funzione che converte logits in probabilità [0,1] |
| Batch | Gruppo di campioni processati insieme |
| Epoch | Un passaggio completo su tutto il dataset di training |
| Step / Iteration | Un aggiornamento dei pesi = un batch processato |
| Gradient | Derivata della loss rispetto a ciascun peso |
| Backward pass | Calcolo automatico dei gradienti (chain rule) |
| Checkpoint | File `.pth` che contiene lo stato del modello |
| State dict | Dizionario che mappa nomi dei layer → tensori dei pesi |
| Fine-tuning | Partire da pesi pretrained e continuare il training |
| Transfer learning | Riusare un modello addestrato per un task A su un task B correlato |
| Domain gap | Differenza tra dominio di addestramento e di test |
| Domain transfer | Tecnica per ridurre il domain gap |
| Overfitting | Il modello memorizza il train ma non generalizza al val |
| Class imbalance | Una o più classi sono molto più rare delle altre |
| Class weights | Pesi nella loss per dare più importanza alle classi rare |
| Data augmentation | Trasformazioni casuali per aumentare varietà del training |
| Selection bias | Bias introdotto scegliendo il modello in base a una metrica |
| GAN | Generative Adversarial Network (Generator + Discriminator) |
| Conditional GAN | GAN dove D vede anche l'input, non solo l'output |
| PatchGAN | Discriminator che produce score map invece di scalare |
| Adversarial loss | Componente della loss del GAN che spinge G a ingannare D |
| L1 loss | Loss "fedeltà": somma dei valori assoluti delle differenze pixel-per-pixel |
| Koschmieder model | Modello fisico per scattering atmosferico (nebbia) |
| ConcatDataset | PyTorch utility per unire più Dataset in uno solo |
| INTER_NEAREST | Interpolazione nearest-neighbor (per maschere discrete) |
| mIoU (mean IoU) | Metrica principale di segmentation, media degli IoU per classe |

---

## 18. Come esercitarsi prima dell'esame

1. **Memorizza i 7 numeri principali**:
   - U-Net v2: test clean 0.7168, foggy medium 0.6652, foggy dense 0.5377
   - U-Net v3: test clean 0.7264, foggy medium 0.7076, foggy dense 0.6866
   - Recupero: 82% medium, 83% dense

2. **Saper raccontare la storia in 5 atti** (sezione 15) senza guardare. Pratica ad alta
   voce, davanti allo specchio o registrandoti.

3. **Disegnare a mano la pipeline end-to-end**:
   ```
   VDD clean → train U-Net (Fase 1) → U-Net v2 (mIoU 0.72)
                                          ↓ test su VDD foggy (Fase 4)
                                          → calo (-7%, -25%)
   Foggy Cityscapes paired → train Pix2Pix (Fase 2) → 2 G addestrati
                                                          ↓ apply a VDD (Fase 3)
                                                          → VDD foggy
   VDD clean + VDD foggy → re-train U-Net (Fase 5) → U-Net v3
                                                       ↓ test
                                                       → recupero ~82-83%
   ```

4. **Rispondere alle 25 domande della sezione 16** ad alta voce. Se non sai rispondere
   a una, è quella su cui devi tornare.

5. **Aprire ogni file del progetto e spiegarselo**: se riesci a raccontarlo a una persona
   non tecnica, lo sai davvero.

6. **Letture consigliate**:
   - Paper U-Net (Ronneberger 2015) — 5 pagine, molto chiaro
   - Paper Pix2Pix (Isola 2017) — sezioni 1-3
   - Paper Foggy Cityscapes (Sakaridis 2018) — solo abstract per il contesto
   - Paper VDD (2024) — abstract + sezione dataset

---

*Documento aggiornato il 9 maggio 2026, dopo il completamento di tutte le 5 fasi del
progetto. Risultati definitivi: U-Net v2 (test mIoU 0.7168 clean, calo -7%/-25% sotto
nebbia) → U-Net v3 (recupero 82-83% via mixed training). Resta solo il report scritto.*

**Numeri da ricordare**: 0.7168, 0.6652, 0.5377 (v2) / 0.7264, 0.7076, 0.6866 (v3) / 82%, 83% recovery.
