# Codice spiegato

**Per**: navigare il codebase del progetto, file per file, sapendo cosa fa ogni pezzo
**Stato**: aggiornato a Fase 5 completata (9 maggio 2026)

Questo documento accompagna il codice. L'idea è: apri il file VS Code, leggi qui la sezione corrispondente, capisci scelte e dettagli. Combinato con `guida_studio.md` (concetti) hai il quadro completo.

---

## Struttura del repository

```
fog-uav-robustness/
├── data/
│   ├── raw/
│   │   ├── VDD/VDD/                  # dataset originale (in locale e Drive)
│   │   └── foggy_cityscapes/         # dataset GAN training
│   └── processed/
│       ├── VDD_foggy_medium_768/     # generato in Fase 3
│       └── VDD_foggy_dense_768/
├── docs/
│   ├── guida_studio.md               # concetti
│   ├── codice_spiegato.md            # questo file
│   └── fase1_completata.md
├── notebooks/                        # Colab notebooks (uno per fase)
│   ├── 01_train_unet_clean.ipynb
│   ├── 02_train_pix2pix.ipynb
│   ├── 03_generate_foggy_vdd.ipynb
│   ├── 04_evaluate_foggy_robustness.ipynb
│   └── 05_retrain_unet_mixed.ipynb
├── outputs/
│   ├── runs/                         # checkpoint, history.json, tb logs
│   └── figures/                      # PNG per il report
├── scripts/                          # script standalone (smoke test, viz)
│   ├── check_env.py
│   ├── visualize_vdd.py
│   ├── visualize_foggy_cityscapes.py
│   ├── test_dataloader.py
│   ├── test_model.py
│   └── test_pix2pix.py
├── src/
│   ├── datasets/
│   │   ├── vdd.py                    # VDDDataset
│   │   └── foggy_cityscapes.py       # FoggyCityscapesPairedDataset
│   ├── models/
│   │   ├── unet.py                   # build_unet (smp wrapper)
│   │   └── gan/
│   │       ├── __init__.py
│   │       └── pix2pix.py            # Generator + Discriminator
│   ├── training/
│   │   ├── train_unet.py             # training U-Net (Fase 1 + Fase 5 mixed)
│   │   └── train_pix2pix.py          # training GAN (Fase 2)
│   ├── inference/
│   │   └── generate_foggy_vdd.py     # applica G a VDD (Fase 3)
│   ├── evaluation/
│   │   ├── metrics.py                # SegmentationMetrics
│   │   └── evaluate.py               # standalone eval, JSON output
│   └── utils/
│       └── transforms.py             # augmentations Albumentations
├── README.md
├── requirements.txt
└── requirements-colab.txt
```

---

# PARTE A — Codice Fase 1 (U-Net su VDD clean)

## 1. `src/datasets/vdd.py` — Dataset VDD

**Classe**: `VDDDataset(Dataset)` — implementa il protocollo PyTorch.

### Cosa fa

1. **`__init__`**: scansiona `<root>/<split>/src/*.JPG`, verifica che ogni JPG abbia la
   maschera corrispondente in `<root>/<split>/gt/<stem>.png`. Salva la lista dei path.
2. **`__len__`**: restituisce il numero di immagini.
3. **`__getitem__(idx)`**:
   - Legge l'immagine col `cv2.imread()` (BGR), converte a RGB
   - Legge la maschera con `cv2.IMREAD_UNCHANGED` (mantiene i class ID)
   - Se c'è un `transform` (Albumentations): applica e restituisce tensori
   - Altrimenti: converte a tensori manualmente (image float, mask long)

### Detail importante: BGR → RGB

OpenCV per motivi storici carica BGR. Se non convertiamo, il modello pretrained (che si
aspetta RGB) dà predizioni pessime.

### Helper utile

`get_class_distribution()` itera su tutte le maschere e conta i pixel per ogni classe.
Restituisce `{class_name: frequency}`. Usato in `train_unet.py` per calcolare le
class_weights.

### Le 7 classi VDD

```python
VDD_CLASSES = {
    0: "other", 1: "wall", 2: "road", 3: "vegetation",
    4: "vehicle", 5: "roof", 6: "water"
}
```

---

## 2. `src/utils/transforms.py` — Augmentation Albumentations

Due funzioni pubbliche:

### `get_train_transform(image_size)`

```python
A.Compose([
    A.Resize(image_size, image_size),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
    A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ToTensorV2(),
])
```

Albumentations applica automaticamente la stessa trasformazione spaziale (resize, flip,
rotate) sia all'image che alla mask, ma le trasformazioni di colore (brightness, hue)
solo all'image. Questa "augmentation paired" è gestita dietro le quinte.

### `get_eval_transform(image_size)`

Solo `Resize + Normalize + ToTensorV2`. **Zero casualità** per validazione e test
deterministici.

### Perché Normalize con stats di ImageNet

Il backbone ResNet-34 è pretrained su ImageNet. Per avere predizioni coerenti, le nostre
immagini devono assomigliare statisticamente a quelle di ImageNet:
```
mean = (0.485, 0.456, 0.406)  # R, G, B
std  = (0.229, 0.224, 0.225)
```

---

## 3. `src/models/unet.py` — wrapper U-Net

```python
def build_unet(num_classes=7, encoder_name="resnet34", encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=num_classes,
        activation=None,  # output raw logits, NO softmax
    )
```

`segmentation_models_pytorch` (smp) ci dà la U-Net "pronta". Output: `(B, num_classes, H, W)`
logit grezzi. Niente softmax interno (lo fa CrossEntropyLoss internamente).

Helper `count_parameters()` e `human_readable()` per stampare "24.4M params" invece di
"24436659 params".

---

## 4. `src/evaluation/metrics.py` — SegmentationMetrics

Classe che incapsula `torchmetrics`:
- `MulticlassJaccardIndex(num_classes, average="macro")` → mIoU
- `MulticlassF1Score(num_classes, average="macro")` → F1
- `MulticlassAccuracy(num_classes, average="micro")` → accuracy pixel-wise
- Per classe: `MulticlassJaccardIndex(num_classes, average=None)` → IoU di ogni classe

Tre metodi:
- `update(logits, targets)`: aggiorna stati interni con un batch
- `compute()`: restituisce `{"mIoU", "F1", "accuracy", "per_class_iou": {...}}`
- `reset()`: azzera (chiamare ad inizio di ogni validation epoch)

torchmetrics gestisce internamente l'accumulo cross-batch (somma TP/FP/FN globali) →
risultati corretti anche per validation in mini-batch.

---

## 5. `src/training/train_unet.py` — training script

Lo script principale della Fase 1 e della Fase 5 (con `--data-roots`).

### Argparse principali

```
--data-root PATH                  # path singolo (Fase 1, baseline)
--data-roots PATH [PATH ...]      # lista (Fase 5, mixed training) [NEW]
--image-size 768                   # default 512, ma v2/v3 usano 768
--epochs 30
--batch-size 4
--lr 1e-4
--weight-decay 1e-4
--grad-clip 1.0
--class-weights {none,inverse,inverse_sqrt}   # v2/v3 usano inverse_sqrt
--encoder resnet34
--num-classes 7
```

### Flow principale

```python
# 1. Build datasets
train_transform = get_train_transform(args.image_size)
val_transform = get_eval_transform(args.image_size)

# Fase 5: --data-roots crea ConcatDataset
train_roots = args.data_roots if args.data_roots else [args.data_root]
train_ds = build_train_dataset(train_roots, ..., transform=train_transform)
val_ds = VDDDataset(args.data_root, "val", transform=val_transform)

# 2. Build loaders
train_loader = DataLoader(train_ds, shuffle=True, ...)
val_loader = DataLoader(val_ds, shuffle=False, ...)

# 3. Build model + loss + optimizer + scheduler
model = build_unet(num_classes=7, encoder_name="resnet34")

if args.class_weights != "none":
    # calcola pesi da get_class_distribution()
    weights = 1/sqrt(freqs); weights /= weights.mean()
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
else:
    criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(T_max=epochs)

# 4. Training loop
for epoch in range(1, epochs+1):
    train_loss = train_one_epoch(...)
    val_loss, val_metrics = validate(...)
    scheduler.step()

    if val_metrics["mIoU"] > best_miou:
        torch.save({...}, "best.pth")

# 5. Final saves
torch.save(..., "last.pth")
```

### Detail: `build_train_dataset` (Fase 5)

```python
def build_train_dataset(data_roots, image_size, subset, transform):
    individual = []
    for root in data_roots:
        ds = VDDDataset(root, "train", transform=transform)
        individual.append(ds)
    if len(individual) == 1:
        return individual[0]
    return ConcatDataset(individual)
```

`ConcatDataset` è una utility PyTorch che fa "appare come un unico dataset": `len()`
restituisce la somma, `__getitem__` redirige al dataset giusto. Combinato con
`shuffle=True` nel DataLoader, mescola clean + foggy_medium + foggy_dense in ogni batch.

### Detail: class weights computation

Per evitare di iterare sulle maschere foggy (sono identiche a quelle clean), calcoliamo
la distribuzione dal **primo root** (tipicamente clean):

```python
base_ds = VDDDataset(train_roots[0], "train", transform=None)
dist = base_ds.get_class_distribution()
freqs = np.array([dist.get(VDD_CLASSES[i], 1e-9) for i in range(num_classes)])
```

### Logging TensorBoard

Per ogni epoch salva:
- `train/loss_epoch`, `train/lr`
- `val/loss`, `val/mIoU`, `val/F1`, `val/accuracy`
- `val/iou_<class>` per ogni classe

`history.json` contiene tutto in formato JSON per analisi post-training.

---

## 6. `src/evaluation/evaluate.py` — script standalone di evaluation

Carica un checkpoint e valuta su uno split. **Non fa training, solo forward + metriche**.

### Argparse

```
--checkpoint PATH        # best.pth da valutare
--data-root PATH         # dataset (clean, foggy_medium, foggy_dense)
--split {train,val,test} # default test
--image-size 768
--batch-size 4
--output PATH            # salva JSON con risultati
--no-tb                  # NON scrivere su TensorBoard (utile per le 4 evaluation foggy)
```

### Output JSON

```json
{
  "checkpoint": "...",
  "data_root": "...",
  "split": "test",
  "loss": 0.4204,
  "mIoU": 0.7168,
  "F1": 0.8279,
  "accuracy": 0.8624,
  "per_class_iou": {
    "other": 0.5763, "wall": 0.6173, "road": 0.6718,
    "vegetation": 0.8737, "vehicle": 0.5569,
    "roof": 0.8176, "water": 0.9039
  }
}
```

### Logging TensorBoard (opzionale)

Se `--no-tb` non è passato, scrive in `<run_dir>/tb/` come scalar `test/mIoU`, ecc. Così
nei dashboard TB vedi side-by-side il train e l'eval finale.

---

# PARTE B — Codice Fase 2 (Pix2Pix)

## 7. `src/datasets/foggy_cityscapes.py` — dataset paired

**Classe**: `FoggyCityscapesPairedDataset(Dataset)`.

### Cosa fa

Restituisce coppie `(clean, foggy)`:
- Legge `No_Fog/<id>.png` come input
- Legge `Medium_Fog/<id>.png` o `Dense_Fog/<id>.png` come target

### Split deterministico

Il dataset Foggy Cityscapes non ha split ufficiale. Lo facciamo noi:
- 90% train, 10% val
- Deterministico: ordiniamo i nomi file e prendiamo i primi N% (no random)
- Stesso split per medium e dense (a parità di `--split`)

### Normalizzazione tanh

```python
img = img.astype(np.float32) / 255.0  # [0, 1]
img = (img - 0.5) / 0.5                # [-1, +1]
```

Il Pix2Pix ha output `tanh` ([-1, +1]), quindi anche i target devono essere lì.

### Augmentation paired

Albumentations supporta `additional_targets={"foggy": "image"}`: la stessa
trasformazione spaziale (flip, rotate) viene applicata a entrambe `clean` e `foggy`.
Senza questa configurazione, le coppie non sarebbero più allineate.

### Helper `denormalize_tanh`

Inverte la normalizzazione `[-1, +1] → [0, 1]`. Usato in
`train_pix2pix.py` per visualizzare gli output durante il training.

---

## 8. `src/models/gan/pix2pix.py` — Generator + Discriminator

### `Pix2PixGenerator`

```python
class Pix2PixGenerator(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet",
                 in_channels=3, out_channels=3):
        super().__init__()
        self.unet = smp.Unet(encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              in_channels=in_channels,
                              classes=out_channels,
                              activation=None)

    def forward(self, x):
        out = self.unet(x)
        return torch.tanh(out)  # [-1, +1]
```

Riusa `smp.Unet` con `classes=3` (output RGB, NON 7 classi). Aggiunge `tanh` finale.
~24.44M parametri (encoder pretrained, decoder + tanh head from scratch).

### `PatchGANDiscriminator`

70×70 PatchGAN classico. 5 strati conv 4×4:
```
Conv(6→64) → LeakyReLU
Conv(64→128) → BN → LeakyReLU
Conv(128→256) → BN → LeakyReLU
Conv(256→512) → BN → LeakyReLU
Conv(512→1)
```

Input 6 canali = `cat([clean, foggy], dim=1)` → conditional GAN.
Con input 256×256 → output 30×30 score map (logit, no sigmoid).
~2.77M parametri (9× più piccolo del G).

Init DCGAN-style:
```python
nn.init.normal_(m.weight, mean=0.0, std=0.02)
```

---

## 9. `src/training/train_pix2pix.py` — training GAN

### Loss

```python
bce = nn.BCEWithLogitsLoss()  # numericamente stabile
l1 = nn.L1Loss()
```

### Loop principale (per batch)

```python
# Update D
with torch.no_grad():
    fake_foggy = G(clean)
d_real = D(clean, real_foggy)
d_fake = D(clean, fake_foggy)
loss_D_real = bce(d_real, ones_like(d_real))
loss_D_fake = bce(d_fake, zeros_like(d_fake))
loss_D = 0.5 * (loss_D_real + loss_D_fake)
opt_D.zero_grad(); loss_D.backward(); opt_D.step()

# Update G
fake_foggy = G(clean)  # con grad
d_fake_for_G = D(clean, fake_foggy)
loss_G_adv = bce(d_fake_for_G, ones_like(d_fake_for_G))
loss_G_l1 = l1(fake_foggy, real_foggy)
loss_G = loss_G_adv + 100.0 * loss_G_l1   # λ_L1 = 100
opt_G.zero_grad(); loss_G.backward(); opt_G.step()
```

### Detach trick

Quando aggiorniamo D, `fake_foggy` è generato senza grad (`.no_grad()` o `.detach()`).
Così `loss_D.backward()` non aggiorna i pesi di G (che è "in pausa" mentre alleniamo D).

### Optimizers

```python
opt_G = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
```

`betas=(0.5, 0.999)` è il default Pix2Pix (più stabile per GAN).

### Best checkpoint criterion

Usiamo **val L1** del Generator (no D involved). Bassa L1 → il fake è simile al real.

```python
val_l1 = sum(L1(G(clean), real_foggy)) / num_batches_val
if val_l1 < best_val_l1:
    best_val_l1 = val_l1
    torch.save({"G_state_dict": G.state_dict(), ...}, "G_best.pth")
```

### Sample grid logging

Ogni 5 epoch salviamo un grid `clean | fake | real` di 4 immagini di val. Su TensorBoard
e su disco (`samples/epoch_NNN.png`). Permette di vedere visivamente come migliora la
qualità nel tempo.

---

# PARTE C — Codice Fase 3 (Generazione VDD foggy)

## 10. `src/inference/generate_foggy_vdd.py` — applica G a tutto VDD

### Argparse

```
--generator PATH         # G_best.pth da caricare
--vdd-root PATH          # data/raw/VDD/VDD
--output-root PATH       # data/processed/VDD_foggy_medium_768
--apply-size 768         # risoluzione di applicazione
--save-size 768          # risoluzione di salvataggio
--batch-size 4
--image-format jpg
--jpeg-quality 95
```

### Flow

```
Per ogni split (train, val, test):
    Per ogni batch di immagini:
        1. cv2.imread originale BGR (4000×3000)
        2. preprocess_batch:
           - BGR → RGB
           - resize a apply_size
           - normalize (-1, +1) tanh
           - stack in tensor (B, 3, apply_size, apply_size)
        3. y = G(x)
        4. postprocess_batch:
           - clamp(-1, +1), denormalize a [0, 255]
           - permute CHW → HWC
           - resize a save_size se diverso
           - RGB → BGR
        5. Per ogni immagine:
           - cv2.imwrite foggy a save_size in JPG q=95
           - cv2.imread mask originale (4000×3000)
           - resize_mask con cv2.INTER_NEAREST a save_size  ← KEY!
           - cv2.imwrite mask resized in PNG
```

### Bug critico (e suo fix)

**Errore iniziale**: copiavo le maschere così com'erano (4000×3000) mentre le immagini
foggy erano salvate a 768×768. Albumentations richiede shape uguali → crash a runtime
in Fase 4.

**Fix**: `resize_mask()` con `cv2.INTER_NEAREST`:

```python
def resize_mask(mask, save_size):
    if mask.shape[0] == save_size and mask.shape[1] == save_size:
        return mask
    return cv2.resize(mask, (save_size, save_size), interpolation=cv2.INTER_NEAREST)
```

**Perché NEAREST e non BILINEAR?** Le maschere contengono class ID discreti (0..6).
Bilineare produrrebbe valori intermedi (es. 2.5) **invalidi**. NEAREST prende il valore
del pixel più vicino senza fare medie, preservando i class ID.

### Loading del checkpoint

```python
def load_generator(ckpt_path, encoder_name, device):
    G = Pix2PixGenerator(
        encoder_name=encoder_name,
        encoder_weights=None,  # NON ricaricare ImageNet, useremo i pesi del ckpt
        in_channels=3, out_channels=3
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    G.load_state_dict(ckpt["G_state_dict"])
    G.eval()
    return G
```

### Performance

T4: ~194 ms/immagine, 400 immagini in ~78 secondi totali.

---

# PARTE D — Codice Fasi 4+5 (Robustezza e recupero)

## 11. `notebooks/04_evaluate_foggy_robustness.ipynb`

29 celle, 11 sezioni. Usa solo codice esistente (`evaluate.py`), nessun nuovo file Python.

### Sezione chiave: la cella 8 (riscritta con il fix)

Una unica cella con 7 step in sequenza:
- **A**: clean up VDD locale corrotto, ricopia da Drive
- **B**: rimuovi i 4 dataset foggy locali (vecchi con maschere 4000×3000)
- **C**: restore Generator da Drive
- **D**: pip install requirements
- **E**: rigenera i 4 dataset foggy con il fix delle maschere
- **F**: verify mask shapes (assert 768×768)
- **G**: backup foggy regenerated su Drive

### La tabella aggregata (cella 24)

```python
test_sets = [
    ('clean', 'test_results_clean.json'),
    ('medium @256', 'test_results_foggy_medium_256.json'),
    ('medium @768', 'test_results_foggy_medium_768.json'),
    ('dense @256', 'test_results_foggy_dense_256.json'),
    ('dense @768', 'test_results_foggy_dense_768.json'),
]
```

Calcola Δ vs clean baseline e per-class IoU comparison.

---

## 12. `notebooks/05_retrain_unet_mixed.ipynb`

27 celle, 11 sezioni. Usa il `train_unet.py` aggiornato con `--data-roots`.

### Cella chiave: la cella di training (sezione 7)

```bash
!python src/training/train_unet.py \
    --data-root /content/data/VDD \
    --data-roots /content/data/VDD \
                 /content/data/VDD_foggy_medium_768 \
                 /content/data/VDD_foggy_dense_768 \
    --output-dir outputs/runs \
    --run-name unet_resnet34_mixed_v3 \
    --image-size 768 \
    --epochs 30 \
    --batch-size 4 \
    --num-workers 2 \
    --lr 1e-4 \
    --weight-decay 1e-4 \
    --grad-clip 1.0 \
    --class-weights inverse_sqrt \
    --seed 42
```

Note:
- `--data-root` (singolare) usato per validation (clean only) → benchmark consistente con v2
- `--data-roots` (plurale) per training (3 roots) → ConcatDataset 840 immagini
- Tutti gli altri hyperparams = stessi della v2 → isoliamo l'effetto della data augmentation

### La tabella v2 vs v3 (cella 26)

```python
v2 = ...outputs/runs/unet_resnet34_clean_v2_weighted/test_results_*.json
v3 = ...outputs/runs/unet_resnet34_mixed_v3/test_results_*.json

print('v2 mIoU | v3 mIoU | v3 - v2 | recovered?')
for label in ['clean', 'medium @768', 'dense @768']:
    delta = v3[label]['mIoU'] - v2[label]['mIoU']
    if 'foggy' in label:
        clean_baseline = v2['clean']['mIoU']
        recovery = delta / (clean_baseline - v2[label]['mIoU']) * 100
    print(f"{label:12s}  {v2[label]['mIoU']:.4f}  {v3[label]['mIoU']:.4f}  {delta:+.4f}  {recovery:+.1f}%")
```

### La figura comparativa (cella 27)

`phase5_v2_vs_v3.png` con 3 coppie di barre. Sarà la figura principale del report.

---

# PARTE E — Scripts di utilità

## 13. `scripts/check_env.py`

Stampa Python version, PyTorch version, CUDA disponibile, segmentation_models_pytorch
version, OpenCV, Albumentations. Sanity check dell'environment all'inizio di ogni sessione.

## 14. `scripts/visualize_vdd.py`

Carica N immagini da VDD e crea una figura grid `image | mask | overlay`. Output:
`outputs/figures/vdd_<split>_samples.png`. Utile per verificare che il dataset sia
correttamente caricato.

## 15. `scripts/visualize_foggy_cityscapes.py`

Tre colonne: `clean | foggy | |foggy - clean|`. La terza colonna è la differenza
normalizzata (mostra dove la nebbia "agisce di più"). Mostra anche il `diff_mean`
numerico, indicatore di severità.

## 16. `scripts/test_dataloader.py`

Crea un VDDDataset, fa `next(iter(loader))`, verifica shape, dtype, range. Smoke test
del data pipeline.

## 17. `scripts/test_model.py`

Builda U-Net, fa forward su batch fake, verifica shape output. Smoke test del modello.

## 18. `scripts/test_pix2pix.py`

Builda G + D, fa forward, fa una single training step (D update + G update), verifica
che i gradienti fluiscano. Smoke test del Pix2Pix.

---

## Patterns trasversali

### Pattern 1: smoke test prima di training reale

Tutte le pipeline hanno un `--subset N` o uno smoke test script. Sempre verificare prima
con piccoli dati che le shape siano corrette, la loss scenda, la pipeline non crashi.
Solo poi fare il training reale di ore.

### Pattern 2: --output-dir con run-name

Ogni training crea `outputs/runs/<run_name>/` con:
- `config.json` (hyperparams usati)
- `best.pth` / `last.pth` (checkpoint)
- `history.json` (metriche per epoch)
- `tb/` (TensorBoard events)

Permette di rieseguire o confrontare run multipli.

### Pattern 3: TensorBoard scalar naming

`train/loss_epoch`, `val/mIoU`, `val/iou_<class>` → struttura ad albero che TB raggruppa
visivamente.

### Pattern 4: lavoro Colab + Drive

- `git clone` del progetto in `/content/`
- Mount Drive
- Copia data e checkpoint da Drive a `/content/data/` (SSD locale, più veloce)
- Run training/inference
- Backup risultati su Drive prima della scadenza sessione

Pattern: tutto è ricostruibile da GitHub + Drive in qualsiasi nuova sessione Colab.

---

## Numeri di riferimento

### Tempi training su T4

| Step | Tempo |
|------|-------|
| U-Net v2 (Fase 1, 30 epoch, 280 train) | ~30 min |
| Pix2Pix medium (50 epoch, 450 train) | ~57 min |
| Pix2Pix dense (50 epoch, 450 train) | ~57 min |
| Generate VDD foggy (4 dataset, 400 img) | ~5 min totali |
| Eval su 5 datasets | ~5 min totali |
| U-Net v3 (Fase 5, 30 epoch, 840 train) | ~60 min |

### Dimensioni file

| File | Dimensione |
|------|-----------|
| best.pth U-Net (~24M params) | ~95 MB |
| best.pth Pix2Pix G | ~95 MB |
| Pix2Pix D | ~10 MB |
| VDD_foggy_*_768 (400 img a 768) | ~200 MB ciascuno |

### Pesi class_weights inverse_sqrt (Fase 1 fix)

Sul training set di VDD:
- vegetation: 0.30 (frequente, peso basso)
- roof: 0.45
- road: 0.55
- water: 0.95
- wall: 1.05
- other: 1.50
- vehicle: 4.50 (rara, peso alto)

(media = 1.0 dopo normalizzazione)

---

*Documento aggiornato il 9 maggio 2026, dopo il completamento di tutte le 5 fasi del progetto.*
