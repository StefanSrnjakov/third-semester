# Vaja 5 – Multipla regresija in interpretacija modelov

## Opis projekta

Ta notebook vsebuje celovito analizo multiplih regresijskih modelov za proces 3D tiskanja medicinskih implantatov. Analiza sledi principom Lean Six Sigma (faza Analyze in Improve).

## Vsebina notebook-a

### 📊 Struktura (50 celic)

1. **Priprava podatkov (Celice 1-13)**
   - Uvoz knjižnic
   - Nalaganje podatkov iz `expanded_data.csv`
   - Kreiranje novih spremenljivk (binarne, ordinalne, standardizirane)
   - Vizualizacija podatkov (korelacije, distribucije)

2. **Model 1: Linearna regresija - Trajanje koraka (Celice 14-18)**
   - Napovedovanje `Trajanje (h)`
   - Napovedniki: Čakanje, Je_VA, Napake, Iteracija
   - Preverjanje predpostavk (normalnost, homoskedastičnost, multikolinearnost)
   - Tabela rezultatov z interpretacijami

3. **Model 2: Linearna regresija - Skupni čas (Celice 19-23)**
   - Napovedovanje `Skupni_cas` (Trajanje + Čakanje)
   - Napovedniki: Je_VA, Je_NVA, Napake, Iteracija, Cakanje_ord
   - Preverjanje predpostavk
   - Tabela rezultatov z interpretacijami

4. **Model 3: Logistična regresija - Napake (Celice 24-28)**
   - Napovedovanje `Ima_napako` (binarna spremenljivka)
   - Napovedniki: Trajanje, Čakanje, Je_NVA, Iteracija
   - Confusion matrix in ROC krivulja
   - Tabela rezultatov z Odds Ratios

5. **Model 4: Ordinalna regresija - Kategorija trajanja (Celice 29-33)**
   - Napovedovanje `Trajanje_kategorija` (Kratko/Srednje/Dolgo)
   - Napovedniki: Čakanje, Je_VA, Napake, Cakanje_ord
   - Confusion matrix
   - Tabela rezultatov s threshold vrednostmi

6. **Primerjava in zaključki (Celice 34-37)**
   - Primerjava vseh 4 modelov
   - Interpretacija rezultatov
   - Priporočila za izboljšave (Improve faza)
   - Shranjevanje rezultatov

## 🚀 Uporaba

### Predpogoji

Potrebne knjižnice:
```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn statsmodels openpyxl
```

### Zagon

1. Odprite notebook v Jupyter:
```bash
jupyter notebook Vaja5_Multipla_Regresija.ipynb
```

2. Zaženite vse celice (Run All) ali jih izvajajte postopoma za boljše razumevanje

3. Rezultati bodo shranjeni v `rezultati_regresija.xlsx` (če je openpyxl nameščen)

## 📋 Ključne ugotovitve

### Zgrajeni modeli

| Model | Tip | Ciljna spremenljivka | Št. napovednikov |
|-------|-----|---------------------|------------------|
| Model 1 | Linearna | Trajanje (h) | 4 |
| Model 2 | Linearna | Skupni_cas (h) | 5 |
| Model 3 | Logistična | Ima_napako (0/1) | 4 |
| Model 4 | Ordinalna | Trajanje_kategorija | 4 |

### Preverjene predpostavke

Za vsak model smo preverili:
- ✅ **Normalnost residualov** (Shapiro-Wilk test)
- ✅ **Homoskedastičnost** (Breusch-Pagan test)
- ✅ **Multikolinearnost** (VIF - Variance Inflation Factor)

### Rezultati

Notebook vsebuje:
- **Standardizirane in nestandardizirane koeficiente** za linearne modele
- **Odds ratios** za logistično regresijo
- **Threshold vrednosti** za ordinalno regresijo
- **Statistično značilnost** vseh koeficientov (p-vrednosti)
- **Interpretacije** v slovenskem jeziku

## 📁 Datoteke

- `expanded_data.csv` - Vhodni podatki (3D tiskanje implantatov)
- `Vaja5_Multipla_Regresija.ipynb` - Glavni notebook
- `rezultati_regresija.xlsx` - Izhodni rezultati (po zagonu)
- `README.md` - Ta datoteka

## 🎯 Lean Six Sigma povezava

### Faza Analyze
- Identifikacija ključnih vplivnih faktorjev
- Statistična analiza odnosov med spremenljivkami
- Kvantifikacija učinkov

### Faza Improve
- Priporočila na osnovi statistično značilnih ugotovitev
- Fokus na zmanjšanje čakalnih časov
- Preprečevanje napak
- Optimizacija VA (Value Added) aktivnosti

## 📊 Vizualizacije

Notebook vključuje:
- Korelacijsko matriko
- Histograme distribucij
- Q-Q plote za preverjanje normalnosti
- Residuals plots
- Confusion matrices
- ROC krivulje

## 🔍 Interpretacije

Vsak model vsebuje:
- Podrobno tabelo rezultatov
- Interpretacijo koeficientov v slovenščini
- Statistično značilnost (označeno z ***, **, *)
- Praktične implikacije za proces

## ✨ Priporočila za uporabo

1. **Najprej zaženite vse celice** da dobite celoten pregled
2. **Preglejte predpostavke** za vsak model
3. **Interpretirajte koeficiente** v kontekstu vašega procesa
4. **Primerjajte modele** glede na AIC/BIC in R²
5. **Uporabite ugotovitve** za implementacijo izboljšav

## 📚 Reference

- **Statsmodels:** Za OLS, Logit in OrderedModel regresijo
- **Scikit-learn:** Za dodatne metrike in preprocessing
- **Lean Six Sigma:** DMAIC metodologija (Analyze & Improve faze)

## 👨‍💻 Avtor

Notebook je bil ustvarjen za potrebe vaje 5 pri predmetu Vitko Upravljanje.

---

**Verzija:** 1.0  
**Datum:** November 2025



