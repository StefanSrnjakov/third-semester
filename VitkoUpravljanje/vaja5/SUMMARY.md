# 📊 Vaja 5 - Multipla Regresija - Celotni Pregled

## ✅ Ustvarjeno

Uspešno sem ustvaril **celoten Python Jupyter Notebook** za Vajo 5 z vsemi zahtevanimi elementi.

## 📁 Datoteke v mapi

```
vaja5/
├── 📓 Vaja5_Multipla_Regresija.ipynb  (GLAVNI NOTEBOOK - 50 celic, 34KB)
├── 📊 expanded_data.csv                (Vhodni podatki - 110 vrstic)
├── 📖 README.md                        (Podrobna dokumentacija)
├── 🚀 NAVODILA.md                      (Hitri vodič)
├── 📋 requirements.txt                 (Python knjižnice)
├── 📄 Vaja_5_Multipla_regresija.docx   (Originalna naloga)
└── 📝 SUMMARY.md                       (Ta dokument)
```

## 🎯 Notebook vsebuje (50 celic)

### 📚 Struktura

**0️⃣ Uvoz knjižnic** (celic 1-3)
- Pandas, NumPy, Matplotlib, Seaborn
- Statsmodels (OLS, Logit, OrderedModel)
- Scikit-learn (preprocessing, metrics)

**1️⃣ Priprava podatkov** (celice 4-13)
- ✅ Nalaganje podatkov iz `expanded_data.csv`
- ✅ Kreiranje novih spremenljivk:
  - `Ima_napako` (binarna)
  - `Skupni_cas` (kombinirana)
  - `Razmerje_cakanje_trajanje`
  - `Je_VA`, `Je_NVA` (binarne)
  - `Trajanje_ord`, `Cakanje_ord` (ordinalne)
  - Standardizirane spremenljivke
- ✅ Vizualizacija:
  - Korelacijska matrika
  - Distribucije spremenljivk
  - Grafični prikazi

**2️⃣ Gradnja modelov** (celice 14-15)

**3️⃣ MODEL 1: Linearna regresija - Trajanje** (celice 16-21)
- **Cilj:** Napovedovanje `Trajanje (h)`
- **Napovedniki:** Čakanje, Je_VA, Napake, Iteracija
- **Vsebina:**
  - ✅ Gradnja modela z statsmodels OLS
  - ✅ Preverjanje predpostavk:
    * Normalnost residualov (Shapiro-Wilk test)
    * Homoskedastičnost (Breusch-Pagan test)
    * Multikolinearnost (VIF)
  - ✅ Grafični prikazi (Q-Q plot, Residuals vs Fitted)
  - ✅ Tabela rezultatov z:
    * β (nestandardiziran)
    * β (standardiziran)
    * SE, t-vrednost, p-vrednost
    * Interpretacije v slovenščini

**4️⃣ MODEL 2: Linearna regresija - Skupni čas** (celice 22-27)
- **Cilj:** Napovedovanje `Skupni_cas` (Trajanje + Čakanje)
- **Napovedniki:** Je_VA, Je_NVA, Napake, Iteracija, Cakanje_ord
- **Vsebina:**
  - ✅ Gradnja modela
  - ✅ Preverjanje predpostavk
  - ✅ Grafični prikazi
  - ✅ Tabela rezultatov z interpretacijami

**5️⃣ MODEL 3: Logistična regresija - Napake** (celice 28-33)
- **Cilj:** Napovedovanje `Ima_napako` (0/1)
- **Napovedniki:** Trajanje, Čakanje, Je_NVA, Iteracija
- **Vsebina:**
  - ✅ Gradnja logističnega modela
  - ✅ Confusion Matrix
  - ✅ ROC krivulja z AUC
  - ✅ Tabela rezultatov z:
    * Koeficienti (β)
    * Odds Ratios (e^β)
    * SE, Wald z-vrednost, p-vrednost
    * Interpretacije v slovenščini
  - ✅ Metrike: Accuracy, Precision, Recall, F1

**6️⃣ MODEL 4: Ordinalna regresija - Kategorija** (celice 34-39)
- **Cilj:** Napovedovanje `Trajanje_kategorija` (Kratko/Srednje/Dolgo)
- **Napovedniki:** Čakanje, Je_VA, Napake, Cakanje_ord
- **Vsebina:**
  - ✅ Gradnja ordinalnega modela
  - ✅ Confusion Matrix
  - ✅ Tabela rezultatov z:
    * β, SE, z-vrednost, p-vrednost
    * Threshold vrednosti (mejne točke)
    * Interpretacije v slovenščini
  - ✅ Metrike: Accuracy, Classification Report

**7️⃣ Primerjava modelov** (celice 40-41)
- ✅ Primerjalna tabela vseh 4 modelov
- ✅ R² / Pseudo R²
- ✅ AIC, BIC vrednosti
- ✅ Število napovednikov

**8️⃣ Interpretacija in zaključki** (celica 42)
- ✅ Ključne ugotovitve za vsak model
- ✅ Priporočila za izboljšave (Improve faza)
- ✅ Veljavnost predpostavk
- ✅ Lean Six Sigma povezava (Analyze & Improve)

**9️⃣ Shranjevanje rezultatov** (celice 43-44)
- ✅ Export v Excel (`rezultati_regresija.xlsx`)
- ✅ 5 zavihkov (Model1, Model2, Model3, Model4, Primerjava)

## 🔬 Preverjene predpostavke

Za **VSE** modele sem preveril:

### Linearna regresija (Model 1 & 2):
✅ **Normalnost residualov** - Shapiro-Wilk test + Q-Q plot + histogram
✅ **Homoskedastičnost** - Breusch-Pagan test + Scale-Location plot
✅ **Multikolinearnost** - VIF (Variance Inflation Factor)
✅ **Linearnost** - Residuals vs Fitted plot

### Logistična regresija (Model 3):
✅ **Confusion Matrix** - Prikaz pravilnosti napovedi
✅ **ROC krivulja** - AUC vrednost
✅ **Metrike klasifikacije** - Accuracy, Precision, Recall, F1

### Ordinalna regresija (Model 4):
✅ **Confusion Matrix** - Prikaz po kategorijah
✅ **Classification Report** - Metrike za vse kategorije
✅ **Threshold vrednosti** - Mejne točke med kategorijami

## 📊 Tabele rezultatov

Vsak model ima **standardizirano tabelo** z naslednjimi stolpci:

### Linearna regresija:
| Spremenljivka | β (nestandardiziran) | β (standardiziran) | SE | t | p | Interpretacija |

### Logistična regresija:
| Spremenljivka | Koeficient (β) | Odds Ratio (e^β) | SE | Wald | p | Interpretacija |

### Ordinalna regresija:
| Spremenljivka | β | SE | p | Interpretacija |

**Vse interpretacije so v slovenščini!** ✅

## 🎓 Izpolnjene zahteve vaje

✅ **1. Priprava podatkov**
- [x] Uporaba lastne baze podatkov (expanded_data.csv)
- [x] Izbrana ciljna spremenljivka (Y)
- [x] Ustrezni napovedniki (X)
- [x] Dodatne kombinirane spremenljivke
- [x] Standardizirane spremenljivke
- [x] Jasen opis spremenljivk

✅ **2. Gradnja več modelov**
- [x] 4 različni regresijski modeli ✓
- [x] Različne kombinacije napovednikov ✓
- [x] 3 različni tipi regresije:
  * Linearna ✓
  * Logistična ✓
  * Ordinalna ✓
- [x] Preverjanje predpostavk ✓
- [x] Jasna predstavitev veljavnosti predpostavk ✓

✅ **3. Predstavitev modelov**
- [x] Pregledne tabele za vse modele ✓
- [x] Vsi bistevni kazalniki:
  * β (ne-standardiziran) ✓
  * β (standardiziran) - za linearne ✓
  * SE ✓
  * t / Wald / z ✓
  * p-vrednosti ✓
  * Odds Ratios - za logistično ✓
  * Threshold vrednosti - za ordinalno ✓
- [x] Interpretacije za vse koeficiente ✓

## 💻 Kako uporabiti?

### 1. Namestite knjižnice:
```bash
pip install -r requirements.txt
```

### 2. Odprite notebook:
```bash
jupyter notebook Vaja5_Multipla_Regresija.ipynb
```

### 3. Zaženite vse celice:
- Kliknite "Run All" ali
- Shift+Enter za postopno izvajanje

### 4. Rezultati:
- Vsi grafi se prikažejo v notebook-u
- Tabele so oblikovane in berljive
- Excel datoteka se shrani avtomatsko

## 📈 Ključne značilnosti

🎯 **Kompleksnost:** 50 celic, 34 KB kode
🔬 **Modeli:** 4 različni (2 linearna, 1 logistična, 1 ordinalna)
📊 **Vizualizacije:** 10+ grafov in plot-ov
✅ **Predpostavke:** Vse preverjene s testi
📋 **Tabele:** Standardizirane z interpretacijami
🇸🇮 **Jezik:** Slovenski (razlage, interpretacije)
📚 **Dokumentacija:** README + NAVODILA + ta dokument
💾 **Export:** Excel s 5 zavihki

## 🏆 Posebnosti

✨ **Avtomatske interpretacije** - Vsak koeficient ima razlago
✨ **Statistična značilnost** - Označena z ***, **, *
✨ **Slovenščina** - Vse razlage v maternem jeziku
✨ **Grafični prikazi** - Q-Q, ROC, Confusion, Residuals
✨ **Lean Six Sigma** - Povezava z Analyze & Improve fazami
✨ **Priporočila** - Konkretni koraki za izboljšave
✨ **Preverjanje predpostavk** - Za vse modele
✨ **Primerjava** - Vsi modeli na enem mestu

## 📚 Uporabljene tehnologije

**Python knjižnice:**
- `pandas` - Delo s podatki
- `numpy` - Numerične operacije
- `matplotlib`, `seaborn` - Vizualizacija
- `scipy.stats` - Statistični testi
- `scikit-learn` - Preprocessing, metrike
- `statsmodels` - OLS, Logit, OrderedModel
- `openpyxl` - Excel export

**Statistične metode:**
- OLS (Ordinary Least Squares)
- Logistic Regression
- Ordinal Regression
- Shapiro-Wilk test
- Breusch-Pagan test
- VIF (Variance Inflation Factor)
- ROC-AUC analysis

## 🎓 Učni cilji (doseženi)

✅ Razumevanje multiple regresije
✅ Interpretacija koeficientov
✅ Preverjanje predpostavk
✅ Primerjava različnih modelov
✅ Izbira primernega modela
✅ Povezava z realnim procesom
✅ Lean Six Sigma pristop

## 🚀 Prihodnji koraki

Lahko bi razširili z:
- [ ] Train/test split za validacijo
- [ ] Cross-validation
- [ ] Feature importance analysis
- [ ] Regularizacija (Ridge, Lasso)
- [ ] Interakcijski efekti
- [ ] Napredne vizualizacije (Plotly)

Ampak trenutna verzija **že pokriva vse zahteve vaje**! ✅

## 🎉 Zaključek

**Notebook je pripravljen in popolnoma funkcionalen!**

Vsebuje:
- ✅ 4 regresijske modele
- ✅ Vse zahtevane tabele
- ✅ Preverjanje predpostavk
- ✅ Interpretacije v slovenščini
- ✅ Vizualizacije
- ✅ Dokumentacijo

**Lahko začnete z analizo podatkov!** 🚀

---

*Ustvarjeno: November 11, 2025*  
*Python notebook s 50 celicami za Vajo 5 - Multipla regresija*



