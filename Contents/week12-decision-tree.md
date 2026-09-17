# 第 12 週｜決策樹與隨機森林（Decision Tree & Random Forest）

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 12 週／共 16 週　**時數**：3 小時
> **本週關鍵字**：吉尼不純度、資訊增益、過度配適、集成學習、Bagging、隨機森林、特徵重要性、OOB 誤差
> **使用工具**：Google Colab（Python：pandas、NumPy、scikit-learn、xgboost、shap）

---

## 0. 本週課程時間分配

| 節次 | 時間 | 內容 |
|---|---|---|
| Hour 1 | 60 分鐘 | 理論深度：決策樹分割準則、過度配適、集成學習、隨機森林、特徵重要性 |
| Hour 2 | 60 分鐘 | Python 示範演練：決策樹與隨機森林建模、三方模型比較、超參數調校 |
| Hour 3（前 40 分鐘） | 40 分鐘 | 延伸練習（兩組模擬資料，附完整程式碼）＋ 綜合案例研究 |
| Hour 3（後 20 分鐘） | 20 分鐘 | **碩士論文延伸應用**：兩個可行論文方向之主題定義、方法說明、模擬資料與 Python 實作展示 |

> [!NOTE]
> 建議於 [Google Colab](https://colab.research.google.com/) 開啟新筆記本，依序將本講義程式碼區塊複製貼上執行。本週延續前幾週建立的模擬車輛感測資料集情境。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 說明決策樹的分割準則（吉尼不純度、熵、資訊增益），並理解迴歸樹如何以變異數縮減作為分割依據。
2. 具體理解決策樹的過度配適現象，並運用樹深度等超參數控制模型複雜度。
3. 說明集成學習的核心概念，區分 Bagging 與 Boosting 兩大典範的差異。
4. 建立隨機森林模型，並運用特徵重要性與 OOB 誤差評估模型。
5. 在 Python 中建立決策樹、隨機森林模型，並與第 10 週的線性模型進行三方比較。
6. 說出至少兩個可將本週方法延伸為碩士論文題目的具體方向，並理解其對應的進階方法與實作方式。

---

## Hour 1｜理論深度（60 分鐘）

### 1.0 決策樹導論

第 10 週的線性迴歸與邏輯斯迴歸，都假設應變數與自變數之間存在（線性或對數勝算線性的）**固定函數形式**；本週的決策樹則完全不做這個假設，改以一系列「是／否」問句，將資料反覆切分為越來越純粹的子集合，最終形成一個樹狀的決策規則。這種方法的最大優勢是**高度可解釋**——一棵訓練完成的決策樹，本質上就是一套人類可以逐條閱讀、理解的判斷規則，這在需要向指揮官或非技術人員說明模型邏輯的軍事應用情境中，是相當重要的特性。

---

### 1.1 決策樹分割準則：吉尼不純度與熵

決策樹在每一個節點，都要決定「用哪個特徵、在哪個切點分割」，才能讓分割後的兩個子節點盡可能「純粹」（子節點內的觀測值盡量屬於同一類別）。常用的不純度衡量指標：

**吉尼不純度（Gini Impurity）**：

$$
Gini(D) = 1-\sum_{i=1}^{c} p_i^2
$$

其中 $p_i$ 為節點 $D$ 中屬於第 $i$ 類的觀測值比例， $c$ 為類別數。當節點內所有觀測值皆屬於同一類別時， $Gini=0$（完全純粹）；當各類別比例平均分布時， $Gini$ 達到最大值。

**熵（Entropy）**：

$$
Entropy(D) = -\sum_{i=1}^{c} p_i \log_2 p_i
$$

**資訊增益（Information Gain）**：分割前後熵的減少量，決策樹會選擇資訊增益最大（不純度下降最多）的特徵與切點進行分割：

$$
Gain(D,A) = Entropy(D) - \sum_{v} \frac{|D_v|}{|D|}Entropy(D_v)
$$

其中 $A$ 為候選分割特徵， $D_v$ 為依 $A$ 分割後的各子集合。

> [!NOTE]
> 吉尼不純度與熵在實務上給出的分割結果通常非常相近（兩者都是「不純度」的合理衡量方式），`scikit-learn` 預設使用吉尼不純度（計算速度略快，不需要計算對數）。除非有特殊需求，兩者選擇對最終模型效能的影響通常不大。

---

### 1.2 決策樹回歸：變異數縮減

當應變數為連續數值時，決策樹改以**變異數縮減（Variance Reduction）**作為分割準則——選擇能讓分割後兩個子節點的組內變異數總和最小的切點，其邏輯與 1.1 節的不純度縮減完全一致，只是衡量「不純粹程度」的方式從類別比例換成了數值的變異程度。

$$
\text{變異數縮減} = Var(D) - \sum_v \frac{|D_v|}{|D|}Var(D_v)
$$

---

### 1.3 過度配適：決策樹的核心挑戰

決策樹若不加限制地持續分割，最終能讓**每個葉節點都只包含一筆訓練資料**（完美配適訓練集，訓練誤差為零），但這樣的樹幾乎必然**嚴重過度配適**——它記住了訓練資料的每一個雜訊細節，卻無法有效類推到新的、未見過的資料。

**控制過度配適的常見超參數**：

| 超參數 | 作用 |
|---|---|
| `max_depth` | 限制樹的最大深度 |
| `min_samples_split` | 節點至少要有多少筆資料才允許繼續分割 |
| `min_samples_leaf` | 葉節點至少要保留多少筆資料 |
| `max_leaf_nodes` | 限制葉節點總數 |

> [!IMPORTANT]
> 決策樹的深度與模型表現之間，存在第 9、10 週已多次強調的**偏差—變異取捨（Bias-Variance Trade-off）**：樹太淺，模型過於簡單（高偏差，配適不足）；樹太深，模型記住雜訊（高變異，過度配適）。本週 Hour 2 示範會具體呈現這條經典的 U 型測試集誤差曲線，這也是為什麼幾乎所有樹狀模型的超參數調校，都圍繞著「找到適當的複雜度」這個核心目標。

**決策樹的其他特性**：

| 優點 | 缺點 |
|---|---|
| 高度可解釋，規則可直接閱讀 | 單一決策樹極不穩定：訓練資料些微變動，可能導致樹的結構完全不同 |
| 不需要標準化（分割僅依賴數值大小順序，不受量綱影響） | 容易過度配適，須仰賴剪枝或深度限制 |
| 能自然處理數值與類別型特徵 | 決策邊界為階梯狀，難以捕捉平滑的線性關係 |

---

### 1.4 集成學習導論：Bagging 與 Boosting

決策樹「不穩定」的缺點，恰好催生了機器學習中最重要的思想之一——**集成學習（Ensemble Learning）**：與其仰賴單一模型的判斷，不如訓練**多個**模型，再綜合它們的意見。集成學習有兩大主流典範：

| 典範 | 核心邏輯 | 代表方法 |
|---|---|---|
| **Bagging**（Bootstrap Aggregating） | 平行訓練多個模型，各自使用**獨立抽樣**的訓練資料子集，最終以投票（分類）或平均（迴歸）整合 | 隨機森林 |
| **Boosting** | **依序**訓練多個模型，每一個新模型都刻意針對前一個模型「答錯」的部分加強學習 | 梯度提升樹（如 XGBoost） |

> [!NOTE]
> Bagging 的核心價值是**降低變異**——透過對多個獨立訓練的模型取平均，能有效平滑掉單一模型的隨機雜訊，這正好對症下藥地緩解了 1.3 節提到的決策樹不穩定問題。Boosting 的核心價值則是**降低偏差**——透過持續針對錯誤加強學習，能讓一群原本「很弱」的模型（如淺層決策樹）逐步逼近複雜的真實關係。本週聚焦於 Bagging 陣營的隨機森林，Boosting 方法將在 Hour 3 論文方向一延伸介紹。

---

### 1.5 隨機森林：Bagging ＋ 特徵隨機抽樣

隨機森林在 Bagging 的基礎上，**額外加入一層隨機性**：

1. **樣本隨機（Bootstrap 抽樣）**：從原始訓練集中，以「有放回抽樣」方式，為每一棵樹產生一份大小相同、但內容不完全相同的訓練子集。
2. **特徵隨機**：**每一次節點分割時**，並非考慮全部特徵，而是從**隨機抽取的部分特徵子集**中選出最佳分割——分類問題預設每次隨機抽取約 $\sqrt{p}$ 個特徵（$p$ 為總特徵數），迴歸問題預設約 $p/3$ 個。
3. 重複步驟 1–2，建立 $n$ 棵各自略有不同的決策樹，最終以多數決（分類）或平均（迴歸）整合所有樹的預測。

> [!IMPORTANT]
> 「特徵隨機」這一步是隨機森林相對於單純 Bagging 決策樹的關鍵改良：若不做特徵隨機、只做樣本隨機，當資料中存在一個**特別強的主導特徵**時，幾乎每一棵樹都會優先選用這個特徵作為最頂層的分割依據，導致所有樹的結構高度相似（彼此高度相關），**Bagging 平均多個高度相關模型的變異縮減效果會大打折扣**。強迫每次分割只能看到部分特徵，能有效「打散」樹與樹之間的相關性，讓 Bagging 的變異縮減效益發揮到最大。

---

### 1.6 特徵重要性

隨機森林能自然地量化「每個特徵對預測的貢獻程度」，最常見的是**吉尼重要性（Gini Importance，又稱平均不純度縮減）**：加總某特徵在森林中**所有**用來分割的節點上，帶來的不純度縮減量（並依該節點涵蓋的樣本比例加權），再對所有樹取平均。

> [!CAUTION]
> 吉尼重要性有一個常見的方法論限制：**傾向給予「取值較多、較連續」的特徵較高的重要性**（因為這類特徵有更多可能的切點可供選擇，更容易「偶然」找到能降低不純度的切法），即使該特徵實際上不具真正的預測力。若對此有疑慮，可改用**排列重要性（Permutation Importance）**——隨機打亂某特徵的數值後，觀察模型預測效能下降的幅度，此方法對特徵型態較為公平，但計算成本較高。

---

### 1.7 OOB（Out-of-Bag）誤差估計

由於每棵樹的訓練資料是**有放回抽樣**產生的，平均而言每棵樹大約只會用到原始訓練集中約 63.2% 的觀測值，其餘約 36.8%「沒被抽到」的觀測值稱為該樹的 **OOB（袋外）樣本**。

**OOB 誤差估計的邏輯**：對每一筆訓練資料，只用「沒有用它訓練過」的那些樹進行預測（即該筆資料在這些樹眼中屬於 OOB 樣本），彙整這些「未見過該筆資料的樹」的預測結果，即可得到一個**不需要額外切分驗證集**、卻相當接近真實測試集表現的泛化誤差估計。

$$
P(\text{某筆觀測值成為某棵樹的OOB樣本}) \approx \left(1-\frac{1}{n}\right)^n \xrightarrow{n\to\infty} e^{-1} \approx 0.368
$$

---

### 1.8 方法選擇指南

| 情境 | 建議方法 |
|---|---|
| 需要高度可解釋、規則能直接呈現給非技術人員 | 單一決策樹（限制深度） |
| 追求較佳的預測效能、能接受較低的可解釋性 | 隨機森林 |
| 資料存在複雜非線性關係或特徵交互作用 | 樹狀模型（決策樹／隨機森林）優於線性模型（第 10 週） |
| 資料的真實關係接近線性可加 | 線性／邏輯斯迴歸（第 10 週）可能與樹狀模型表現相當、甚至更好 |
| 需要快速的模型效能初步估計，且資料量不大 | 使用 OOB 誤差，可省去額外切分驗證集 |

### 1.9 常用中英文詞彙對照表

| 中文 | 英文 | 中文 | 英文 |
|---|---|---|---|
| 吉尼不純度 | Gini Impurity | 熵 | Entropy |
| 資訊增益 | Information Gain | 過度配適 | Overfitting |
| 集成學習 | Ensemble Learning | 裝袋法 | Bagging (Bootstrap Aggregating) |
| 提升法 | Boosting | 隨機森林 | Random Forest |
| 特徵重要性 | Feature Importance | 排列重要性 | Permutation Importance |
| 袋外誤差 | Out-of-Bag Error, OOB Error | 梯度提升樹 | Gradient Boosting Tree |

### 1.10 本週公式總表

| 項目 | 公式 |
|---|---|
| 吉尼不純度 | $Gini(D)=1-\sum_i p_i^2$ |
| 熵 | $Entropy(D)=-\sum_i p_i\log_2 p_i$ |
| 資訊增益 | $Gain(D,A)=Entropy(D)-\sum_v\frac{\lvert D_v\rvert}{\lvert D\rvert}Entropy(D_v)$ |
| OOB樣本機率 | $\approx e^{-1}\approx 0.368$ |

### 1.11 各方法的假設條件與限制一覽

| 方法 | 隱含假設 | 主要限制 |
|---|---|---|
| 決策樹 | 無分配假設，資料驅動 | 高度不穩定，容易過度配適 |
| 隨機森林 | 各棵樹之間的誤差應盡量獨立，才能發揮Bagging效益 | 可解釋性遠低於單一決策樹；面對真正線性關係時未必優於線性模型 |
| 吉尼重要性 | 無 | 偏好取值較多的連續特徵，解讀時須留意此限制 |

---

## Hour 2｜Python 示範演練（60 分鐘）

### 2.0 環境設置與模擬資料生成

```python
# ============================================================
# 第12週 Hour 2 示範：決策樹與隨機森林建模
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

!apt-get -qq install fonts-noto-cjk > /dev/null 2>&1
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JT', 'Noto Sans CJK TC']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n = 1000

mileage_km = np.random.normal(45000, 15000, n).clip(500, None)
engine_temp_c = np.random.normal(88, 6, n)
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n)
fuel_efficiency_km_l = 12 - 0.00008*mileage_km + np.random.normal(0, 0.8, n)

# 需求變數的生成方式與第10週相同（真實關係為線性可加），供後續比較線性模型與樹狀模型的表現差異
risk_score = (0.00002*mileage_km + 0.03*engine_temp_c
              + 0.15*vibration_mm_s + np.random.normal(0, 0.5, n))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    'mileage_km': mileage_km.round(1), 'engine_temp_c': engine_temp_c.round(2),
    'vibration_mm_s': vibration_mm_s.round(3), 'fuel_efficiency_km_l': fuel_efficiency_km_l.round(2),
    'needs_maintenance': needs_maintenance,
})
print("資料集形狀:", df.shape)
print(df['needs_maintenance'].value_counts(normalize=True))
```

---

### 2.1 決策樹分類實作與過度配適示範

```python
# ============================================================
# 2.1 決策樹過度配適示範（對照1.3節）
# ============================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df[['mileage_km', 'engine_temp_c', 'vibration_mm_s', 'fuel_efficiency_km_l']]
y = df['needs_maintenance']
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- 不限制深度：展示嚴重過度配適 ---
tree_full = DecisionTreeClassifier(random_state=42)
tree_full.fit(Xtr, ytr)
print(f"不限制深度: 訓練集準確率={accuracy_score(ytr,tree_full.predict(Xtr)):.4f}, "
      f"測試集準確率={accuracy_score(yte,tree_full.predict(Xte)):.4f}, 樹深度={tree_full.get_depth()}")

# --- 系統性比較不同深度下的訓練/測試集表現 ---
depths = [2, 3, 4, 5, 6, 8, 10, 15]
train_accs, test_accs = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(Xtr, ytr)
    train_accs.append(accuracy_score(ytr, t.predict(Xtr)))
    test_accs.append(accuracy_score(yte, t.predict(Xte)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_accs, marker='o', label='訓練集準確率')
plt.plot(depths, test_accs, marker='s', label='測試集準確率')
plt.xlabel('樹深度 max_depth'); plt.ylabel('準確率')
plt.title('決策樹深度 vs. 訓練/測試集準確率（偏差-變異取捨）')
plt.legend(); plt.grid(alpha=0.3)
plt.show()

best_depth = depths[np.argmax(test_accs)]
print(f"\n測試集表現最佳的深度: max_depth={best_depth}, 測試集準確率={max(test_accs):.4f}")
```

**預期輸出**：不限制深度時，訓練集準確率會達到 100%，但測試集準確率僅約 67%（嚴重過度配適）；隨著 `max_depth` 增加，訓練集準確率持續上升，但測試集準確率會在較淺的深度（約 3）就達到峰值，之後隨著深度繼續增加反而逐漸下降——這條經典曲線具體呈現了 1.3 節的偏差—變異取捨：淺層樹尚未配適不足，深層樹則已開始記憶訓練集雜訊。

---

### 2.2 決策樹規則視覺化

```python
# ============================================================
# 2.2 決策樹規則文字化展示（對照1.1節）
# ============================================================
from sklearn.tree import export_text, plot_tree

tree3 = DecisionTreeClassifier(max_depth=3, random_state=42)
tree3.fit(Xtr, ytr)

print("決策樹規則(max_depth=3):")
print(export_text(tree3, feature_names=list(X.columns)))

plt.figure(figsize=(16, 8))
plot_tree(tree3, feature_names=list(X.columns), class_names=['不需要維護','需要維護'],
          filled=True, rounded=True, fontsize=9)
plt.title('決策樹視覺化（max_depth=3）')
plt.show()
```

---

### 2.3 隨機森林實作與特徵重要性

```python
# ============================================================
# 2.3 隨機森林實作（對照1.5節）與特徵重要性（對照1.6節）
# ============================================================
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(Xtr, ytr)
print(f"隨機森林測試集準確率: {accuracy_score(yte, rf.predict(Xte)):.4f}")

importance_df = pd.DataFrame({
    'feature': X.columns, 'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print("\n特徵重要性:")
print(importance_df)

plt.figure(figsize=(7, 4))
plt.barh(importance_df['feature'], importance_df['importance'], color='steelblue')
plt.xlabel('吉尼重要性'); plt.title('隨機森林特徵重要性')
plt.gca().invert_yaxis()
plt.grid(alpha=0.3, axis='x')
plt.show()
```

---

### 2.4 三方模型比較：邏輯斯迴歸 vs. 決策樹 vs. 隨機森林

```python
# ============================================================
# 2.4 三方模型比較（對照第10週邏輯斯迴歸）
# ============================================================
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score

def eval_model(model, name):
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    print(f"{name}: 準確率={accuracy_score(yte,pred):.4f}, F1={f1_score(yte,pred):.4f}, "
          f"AUC={roc_auc_score(yte,proba):.4f}")

eval_model(LogisticRegression(max_iter=1000), "邏輯斯迴歸（第10週）")
eval_model(DecisionTreeClassifier(max_depth=5, random_state=42), "單一決策樹")
eval_model(RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42), "隨機森林")
```

**預期輸出**：隨機森林應明顯優於單一決策樹（驗證集成學習降低變異的效果）；但由於本週模擬資料的真實風險分數是**線性可加**的（呼應第 10 週的資料生成邏輯），**邏輯斯迴歸的表現可能與隨機森林相當、甚至略優**——這是一個重要且誠實的教學發現：**樹狀模型並非在所有情境下都優於線性模型，其優勢主要展現在資料存在非線性關係或特徵交互作用的情境**，本週 Hour 3 論文方向一會進一步探討這個議題。

---

### 2.5 OOB 誤差驗證

```python
# ============================================================
# 2.5 OOB誤差估計（對照1.7節）
# ============================================================

rf_oob = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, oob_score=True)
rf_oob.fit(Xtr, ytr)
print(f"OOB準確率估計: {rf_oob.oob_score_:.4f}")
print(f"實際測試集準確率: {accuracy_score(yte, rf_oob.predict(Xte)):.4f}")
print("\n（OOB誤差不需要額外切分驗證集，卻能相當接近真實測試集表現，")
print("在訓練資料有限、捨不得切分驗證集的情境下特別有用）")
```

---

### 2.6 超參數調校：GridSearchCV

```python
# ============================================================
# 2.6 以GridSearchCV系統性調校隨機森林超參數
# ============================================================
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'min_samples_leaf': [1, 5, 10],
}
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc'
)
grid_search.fit(Xtr, ytr)

print(f"最佳參數組合: {grid_search.best_params_}")
print(f"交叉驗證最佳AUC: {grid_search.best_score_:.4f}")

best_rf = grid_search.best_estimator_
print(f"測試集AUC: {roc_auc_score(yte, best_rf.predict_proba(Xte)[:,1]):.4f}")
```

> [!TIP]
> `GridSearchCV` 會自動嘗試 `param_grid` 中所有參數組合，並以**交叉驗證（Cross-Validation）**（而非單一次的訓練/驗證切分）評估每組參數的表現，是比人工逐一嘗試更嚴謹、更不容易因單次切分的隨機性而誤判最佳參數的標準做法。

---

## Hour 3（前 40 分鐘）｜延伸練習與綜合案例研究

### 3.1 延伸練習一：決策樹迴歸與隨機森林迴歸比較（附完整程式碼）

```python
# ============================================================
# 延伸練習一：彈藥庫月耗損量之決策樹/隨機森林迴歸（對照1.2節）
# ============================================================
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(21)
n1 = 300
storage_temp = np.random.normal(24, 4, n1)
humidity = np.random.normal(50, 10, n1)
storage_age = np.random.uniform(1, 20, n1)
monthly_loss = 5 + 0.8*storage_temp + 0.3*humidity + 1.2*storage_age + np.random.normal(0, 8, n1)

df1 = pd.DataFrame({'storage_temp': storage_temp, 'humidity': humidity,
                      'storage_age': storage_age, 'monthly_loss': monthly_loss})
X1 = df1[['storage_temp', 'humidity', 'storage_age']]
y1 = df1['monthly_loss']
Xtr1, Xte1, ytr1, yte1 = train_test_split(X1, y1, test_size=0.3, random_state=42)

tree_reg = DecisionTreeRegressor(max_depth=4, random_state=42).fit(Xtr1, ytr1)
rf_reg = RandomForestRegressor(n_estimators=200, max_depth=4, random_state=42).fit(Xtr1, ytr1)

print("=== 決策樹迴歸 vs. 隨機森林迴歸 ===")
print(f"決策樹: R2={r2_score(yte1,tree_reg.predict(Xte1)):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(yte1,tree_reg.predict(Xte1))):.2f}")
print(f"隨機森林: R2={r2_score(yte1,rf_reg.predict(Xte1)):.4f}, "
      f"RMSE={np.sqrt(mean_squared_error(yte1,rf_reg.predict(Xte1))):.2f}")
print(f"\n特徵重要性: {dict(zip(X1.columns, rf_reg.feature_importances_.round(3)))}")
```

**詳解**：隨機森林迴歸的 $R^2$ 應明顯優於單一決策樹（如 0.35 vs. 0.16），再次驗證集成學習降低變異的效果；特徵重要性應顯示「儲存年限」是三者中最重要的預測因子。

---

### 3.2 延伸練習二：隨機森林分類於通信裝備故障預測（附完整程式碼）

```python
# ============================================================
# 延伸練習二：通信裝備故障之隨機森林分類
# ============================================================
np.random.seed(31)
n2 = 500
signal = np.random.normal(70, 15, n2)
age = np.random.uniform(1, 60, n2)
usage_hours = np.random.uniform(100, 5000, n2)
logit = -3 + 0.03*age - 0.015*signal + 0.0003*usage_hours
prob = 1/(1+np.exp(-logit))
failure = (np.random.random(n2) < prob).astype(int)

df2 = pd.DataFrame({'signal': signal, 'age': age, 'usage_hours': usage_hours, 'failure': failure})
print(f"故障比例: {df2['failure'].mean()*100:.2f}%")

X2 = df2[['signal', 'age', 'usage_hours']]
y2 = df2['failure']
Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.3, random_state=42, stratify=y2)

rf2 = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42).fit(Xtr2, ytr2)
print(f"準確率={accuracy_score(yte2,rf2.predict(Xte2)):.4f}, "
      f"AUC={roc_auc_score(yte2,rf2.predict_proba(Xte2)[:,1]):.4f}")
print(f"特徵重要性: {dict(zip(X2.columns, rf2.feature_importances_.round(3)))}")
```

**詳解**：三個特徵（訊號強度、裝備年齡、使用時數）的特徵重要性應大致相當，皆對故障預測有實質貢獻，AUC 應落在 0.65–0.75 之間，屬於中等偏佳的區辨能力。

---

### 3.3 綜合案例研究：建立可解釋的裝備故障預測與決策支援系統

> [!NOTE]
> 以下是一個虛構但貼近實務的案例，目的是把本週技術串接起來，示範完整的樹狀模型建模專案思考過程。

**背景**：某聯兵旅保修處希望建立一套裝備故障預測系統，**同時要求模型具備一定的可解釋性**，以便向非技術背景的主管說明預測依據。

**步驟一：以單一決策樹建立初步可解釋規則**（對應 2.1–2.2 節）
先訓練一棵淺層（如 `max_depth=3`）的決策樹，將其規則轉化為可直接對主管說明的判斷邏輯（如「震動值超過 3.54 且里程數超過 5 萬公里者，須列入優先關注名單」）。

**步驟二：以隨機森林提升預測效能**（對應 2.3–2.4 節）
在確認決策樹規則具備基本合理性後，改用隨機森林提升整體預測準確度，並以特徵重要性驗證模型的判斷邏輯與步驟一的決策樹規則大致一致，強化模型的可信度。

**步驟三：以 OOB 誤差與交叉驗證確保模型穩健性**（對應 2.5–2.6 節）
在資料量有限的情境下，善用 OOB 誤差快速評估模型表現，並以 `GridSearchCV` 系統性調校超參數，避免因人工試誤而遺漏更佳的參數組合。

**步驟四：與線性模型比較，確認樹狀模型的必要性**（對應 2.4 節）
在正式導入較難解釋的隨機森林之前，先確認其相對於第 10 週邏輯斯迴歸的效能提升是否顯著——若提升幅度不大，考慮到可解釋性的價值，選用較簡單的線性模型可能才是更合理的決策。

---

## Hour 3（後 20 分鐘）｜碩士論文延伸應用

> [!NOTE]
> 以下兩個方向，示範如何把本週的樹狀模型方法延伸為具備研究貢獻的碩士論文題目——核心邏輯是：**Bagging（隨機森林）與 Boosting（梯度提升樹）是集成學習兩大不同典範，兩者的計算特性與適用情境值得系統性比較；而隨機森林雖然比單一決策樹更強大，但「一群樹的多數決」比「一棵樹的規則」更難向決策者說明，這正是可解釋 AI（Explainable AI）技術的用武之地**。每個方向皆包含主題定義、方法說明、模擬資料設計與完整可執行的 Python 實作。

### 3.4 論文方向一：梯度提升樹與隨機森林於裝備故障預測之比較研究

**主題定義**：如 1.4 節介紹，Bagging（隨機森林）與 Boosting（梯度提升樹）代表集成學習兩種不同的核心邏輯。本研究以 XGBoost 代表 Boosting 陣營，與隨機森林進行系統性比較，並特別聚焦於「超參數調校的計算效率」這個在實務部署中經常被忽略、卻極具意義的比較維度。

**方法說明**：

- **XGBoost（Extreme Gradient Boosting）**：梯度提升樹的高效實作，透過正則化與平行化技巧大幅提升訓練速度，是近年表格型資料競賽中最常見的方法之一。
- **公平比較原則**：兩種方法皆透過 `GridSearchCV` 進行正式的超參數調校（而非隨意指定參數），確保比較的是「兩方法在各自最佳設定下」的表現，而非調校不足的偏頗比較。
- **比較維度**：預測效能（交叉驗證 AUC）與計算效率（調校總耗時）並重，而非僅比較單一維度。

**模擬資料**：模擬 2000 筆裝備感測資料，並刻意加入一個**非線性交互作用**（高里程數「且」高溫同時發生時，故障風險才會大幅上升，而非個別因素的簡單相加）——這種交互作用是樹狀模型相對線性模型最有機會發揮優勢的資料結構。

```python
# ============================================================
# 論文方向一：XGBoost vs. 隨機森林於裝備故障預測之比較
# ============================================================
!pip install xgboost --quiet

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import time

np.random.seed(42)
n = 2000

mileage_km = np.random.normal(45000, 15000, n).clip(500, None)
engine_temp_c = np.random.normal(88, 6, n)
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n)
fuel_efficiency_km_l = 12 - 0.00008*mileage_km + np.random.normal(0, 0.8, n)

# 刻意加入非線性交互作用：高里程"且"高溫同時發生時，風險才會急遽上升
interaction_effect = np.where((mileage_km > 60000) & (engine_temp_c > 92), 6.0, 0.0)
risk_score = (0.000005*mileage_km + 0.005*engine_temp_c + 0.05*vibration_mm_s
              + interaction_effect + np.random.normal(0, 0.5, n))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    'mileage_km': mileage_km.round(1), 'engine_temp_c': engine_temp_c.round(2),
    'vibration_mm_s': vibration_mm_s.round(3), 'fuel_efficiency_km_l': fuel_efficiency_km_l.round(2),
})
X, y = df, needs_maintenance
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

# --- 隨機森林：正式GridSearchCV調校 ---
rf_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    {'n_estimators': [100, 200], 'max_depth': [4, 6, 8]},
    cv=3, scoring='roc_auc'
)
t0 = time.time()
rf_grid.fit(Xtr, ytr)
t_rf = time.time() - t0
print(f"隨機森林: 最佳參數={rf_grid.best_params_}, CV最佳AUC={rf_grid.best_score_:.4f}, "
      f"調校耗時={t_rf:.2f}秒")
print(f"測試集AUC={roc_auc_score(yte, rf_grid.predict_proba(Xte)[:,1]):.4f}")

# --- XGBoost：正式GridSearchCV調校 ---
xgb_grid = GridSearchCV(
    xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
    {'n_estimators': [100, 200], 'max_depth': [2, 3, 4], 'learning_rate': [0.05, 0.1]},
    cv=3, scoring='roc_auc'
)
t0 = time.time()
xgb_grid.fit(Xtr, ytr)
t_xgb = time.time() - t0
print(f"\nXGBoost: 最佳參數={xgb_grid.best_params_}, CV最佳AUC={xgb_grid.best_score_:.4f}, "
      f"調校耗時={t_xgb:.2f}秒")
print(f"測試集AUC={roc_auc_score(yte, xgb_grid.predict_proba(Xte)[:,1]):.4f}")

print(f"\n=== 結論 ===")
print(f"調校速度比較: 隨機森林={t_rf:.2f}秒, XGBoost={t_xgb:.2f}秒, "
      f"XGBoost快{t_rf/t_xgb:.1f}倍")
```

**預期輸出**：兩種方法在經過正式調校後，交叉驗證 AUC 通常相當接近（差距在 0.01 以內），**測試集效能未必是 Boosting 明顯勝出**——這是一個誠實且重要的發現：**在許多實務資料集上，經過妥善調校的 Bagging 與 Boosting 方法效能相近，Boosting 未必總是「更強」**。但 XGBoost 的超參數搜尋耗時通常僅為隨機森林的 **1/4 左右**，這個計算效率上的差異，在需要頻繁重新訓練（如每週更新模型）或大規模資料的實務部署情境下，具有真實且重要的意義。

> [!IMPORTANT]
> 這個結果提醒研究者：**比較機器學習方法時，不應該只看單一維度（如準確率）就宣稱某方法「更好」**。計算效率、訓練穩定性、超參數調校難度、模型可解釋性，都是同樣重要的比較維度，一份嚴謹的碩士論文比較研究，應該對這些維度都有所著墨，而非僅呈現一張「準確率比較表」就下結論。

**論文延伸建議**：可進一步比較 LightGBM（另一種高效的梯度提升實作）與隨機森林、XGBoost 三方的效能與效率；也可以在真正具有更強烈非線性交互作用的資料上重複此比較，探討「交互作用強度」是否會系統性地擴大 Boosting 相對 Bagging 的效能優勢。

---

### 3.5 論文方向二：SHAP 可解釋性分析於軍事裝備維護決策支援之應用

**主題定義**：隨機森林雖然預測效能通常優於單一決策樹，但如 1.3 節提醒，其「數百棵樹多數決」的機制遠比單一決策樹難以解釋——當模型判定某件裝備「需要優先維護」時，維保人員經常需要知道**具體原因**，才能決定實際檢修項目，也才能對模型的判斷建立信任。本研究運用 SHAP（SHapley Additive exPlanations）方法，為隨機森林的預測結果提供個案層級的可解釋分析，驗證此方法能否同時提供全域特徵重要性排序與個別裝備的具體診斷依據。

**方法說明**：

- **SHAP 值的理論基礎**：源自賽局理論中的 Shapley 值，將「模型預測值與基準值（訓練集平均預測值）之間的差距」，公平地分配給每一個特徵，滿足「所有特徵的 SHAP 值加總，恰好等於該筆預測值與基準值的差距」這個嚴謹的數學性質——這正是 SHAP 相對於 1.6 節傳統吉尼重要性最大的優勢：**吉尼重要性只能告訴你「哪個特徵整體而言重要」，SHAP 值能告訴你「對這一筆特定預測，每個特徵各自貢獻了多少」**。
- **全域與個案解釋並重**：對所有測試樣本的 SHAP 值取絕對值平均，可得到與吉尼重要性性質相近的全域特徵排序（可交叉驗證兩者是否一致）；對單一樣本的 SHAP 值，則能提供「為什麼模型認為這件裝備需要維護」的具體診斷。

**模擬資料**：沿用本週 Hour 2 主範例車輛感測資料集與訓練完成的隨機森林模型。

```python
# ============================================================
# 論文方向二：SHAP可解釋性分析於裝備維護決策支援
# ============================================================
!pip install shap --quiet

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import shap

np.random.seed(42)
n = 1000
mileage_km = np.random.normal(45000, 15000, n).clip(500, None)
engine_temp_c = np.random.normal(88, 6, n)
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n)
fuel_efficiency_km_l = 12 - 0.00008*mileage_km + np.random.normal(0, 0.8, n)
risk_score = (0.00002*mileage_km + 0.03*engine_temp_c
              + 0.15*vibration_mm_s + np.random.normal(0, 0.5, n))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    'mileage_km': mileage_km.round(1), 'engine_temp_c': engine_temp_c.round(2),
    'vibration_mm_s': vibration_mm_s.round(3), 'fuel_efficiency_km_l': fuel_efficiency_km_l.round(2),
})
X, y = df, needs_maintenance
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
rf.fit(Xtr, ytr)

# --- 計算SHAP值 ---
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(Xte)
sv = shap_values[:, :, 1] if np.array(shap_values).ndim == 3 else shap_values[1]

# --- 全域特徵重要性：SHAP vs. 傳統吉尼重要性 ---
mean_abs_shap = np.abs(sv).mean(axis=0)
comparison = pd.DataFrame({
    'feature': X.columns,
    'SHAP重要性': mean_abs_shap,
    '吉尼重要性': rf.feature_importances_,
}).sort_values('SHAP重要性', ascending=False)
print("=== 全域特徵重要性比較（SHAP vs. 吉尼重要性）===")
print(comparison)

# --- 個案解釋：找一筆被預測為"需要維護"的樣本，具體說明原因 ---
idx = np.where(rf.predict(Xte) == 1)[0][0]
print(f"\n=== 個案解釋（第{idx}筆測試樣本）===")
print(f"特徵值: {Xte.iloc[idx].to_dict()}")
print(f"預測結果: {'需要維護' if rf.predict(Xte)[idx]==1 else '不需要維護'}, "
      f"機率={rf.predict_proba(Xte)[idx,1]:.4f}")
print(f"\n各特徵SHAP貢獻值（正值代表提高風險預測，負值代表降低）:")
for feat, val in zip(X.columns, sv[idx]):
    print(f"  {feat}: {val:+.4f}")
base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value
print(f"\n基準值(base value) = {base_value:.4f}")
print(f"基準值 + 各特徵SHAP貢獻總和 = {base_value + sv[idx].sum():.4f}（應等於預測機率，驗證可加性）")
```

**預期輸出**：全域特徵重要性排序上，SHAP 與傳統吉尼重要性應給出**一致**的排序（如震動值 > 里程數 > 引擎溫度 > 油耗效率），交叉驗證了兩種方法在「整體而言哪個特徵重要」這個問題上的結論相符；但 SHAP 額外提供的**個案解釋**，能具體呈現「這一件裝備之所以被判定需要維護，是因為震動值貢獻了 +0.29、引擎溫度貢獻了 +0.16……」，且驗證「基準值加上所有特徵的 SHAP 貢獻總和，恰好等於該筆預測機率」——這個可加性驗證，正是 SHAP 值嚴謹數學性質最直觀的展現。

> [!IMPORTANT]
> 這對軍事裝備維護決策支援系統具有直接的應用價值：**維保人員不需要只看到模型丟出「需要維護」四個字，而是能同時看到具體的診斷依據**（如「主要因為震動值異常偏高」），據此決定應該優先檢查哪個子系統，也讓模型的判斷更容易被現場人員信任與採納——這正是可解釋 AI 技術在高風險、高問責情境（軍事、醫療、金融）中日益受到重視的根本原因。

**論文延伸建議**：可進一步將 SHAP 分析應用於本週論文方向一比較的 XGBoost 模型，探討可解釋性方法是否也能同樣有效地應用於 Boosting 陣營的模型；也可以蒐集實際維保人員對「純預測結果」與「附帶 SHAP 解釋的預測結果」兩種呈現方式的信任度與採用意願，進行使用者研究，量化可解釋性對模型實務採用率的實際影響——這是可解釋 AI 領域從「技術可行」走向「實務有效」的重要研究缺口。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1：決策樹為什麼不需要像線性迴歸、K-Means 那樣事先標準化？**
> A：因為決策樹的分割邏輯只依賴數值的**相對大小順序**（例如「震動值是否大於 3.54」），而非數值之間的距離或線性組合。將某特徵的所有數值乘以任意正數（相當於改變量綱），並不會改變資料點之間的相對大小順序，因此也不會改變決策樹的分割結果。這是樹狀模型相對於線性模型、距離基礎方法（如 K-Means、KNN）的一大實務便利之處。

> [!NOTE]
> **Q2：隨機森林的樹越多越好嗎？**
> A：增加樹的數量（`n_estimators`）一般不會導致過度配適加劇（因為每棵樹都是獨立訓練後取平均，更多獨立樣本只會讓平均值更穩定，不會讓模型「記住」更多雜訊），但會增加計算成本，且效益會**遞減**——超過某個數量後，繼續增加樹的數量對效能提升幫助有限。實務上通常在 100–500 棵樹之間，搭配交叉驗證觀察效能是否已經穩定。

> [!NOTE]
> **Q3：隨機森林比單一決策樹一定表現更好嗎？**
> A：在絕大多數情境下是的（這是集成學習的理論保證：多個獨立誤差的平均，變異必然小於等於單一誤差），但如 2.4 節示範，「隨機森林優於決策樹」不代表「樹狀模型優於線性模型」——若資料的真實關係本身就接近線性可加，第 10 週的邏輯斯迴歸可能表現相當、甚至更好，選擇方法時應該根據資料特性判斷，而非預設複雜模型一定優於簡單模型。

> [!NOTE]
> **Q4：Bagging 與 Boosting，哪一個比較不容易過度配適？**
> A：一般而言 Bagging（隨機森林）較不容易過度配適，因為其設計目的就是降低變異；Boosting 由於持續針對錯誤加強學習，若不加以控制（如限制樹的數量、加入正則化項），理論上更容易過度配適，這也是為什麼 XGBoost 等現代 Boosting 實作，都內建了多種正則化機制與早停（Early Stopping）功能。

> [!NOTE]
> **Q5：SHAP 值與吉尼重要性的結論如果不一致，該相信哪一個？**
> A：兩者衡量的其實是略有不同的概念，不一致不代表誰「錯」——吉尼重要性反映「該特徵在樹的建構過程中，平均而言降低了多少不純度」；SHAP 值反映「該特徵對個別預測結果的實際貢獻」，兩者在特徵之間存在強相關、或資料分布不均勻時，確實可能給出不同的排序。若研究目的是解釋個別預測（如向決策者說明單一案例），應優先採用 SHAP；若只需要快速掌握整體特徵的相對重要性，吉尼重要性計算成本更低、也已足夠。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| 決策樹分類 | `DecisionTreeClassifier` | `sklearn.tree` |
| 決策樹迴歸 | `DecisionTreeRegressor` | `sklearn.tree` |
| 決策樹規則文字化 | `export_text` | `sklearn.tree` |
| 決策樹視覺化 | `plot_tree` | `sklearn.tree` |
| 隨機森林分類 | `RandomForestClassifier` | `sklearn.ensemble` |
| 隨機森林迴歸 | `RandomForestRegressor` | `sklearn.ensemble` |
| 超參數網格搜尋 | `GridSearchCV` | `sklearn.model_selection` |
| 梯度提升樹 | `XGBClassifier` | `xgboost` |
| SHAP可解釋性分析 | `TreeExplainer` | `shap` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「吉尼不純度」 vs 「熵」 | 兩者皆衡量節點的不純粹程度，計算方式不同但實務效果相近，`sklearn`預設使用吉尼不純度 |
| 「Bagging」 vs 「Boosting」 | Bagging平行訓練、獨立抽樣，目標降低變異；Boosting依序訓練、針對錯誤加強，目標降低偏差 |
| 「吉尼重要性」 vs 「SHAP值」 | 前者是全域、單一數值的特徵排序；後者能同時提供全域排序與個案層級的具體貢獻值，且具備嚴謹的可加性數學性質 |
| 「OOB誤差」 vs 「交叉驗證誤差」 | 兩者都是不依賴額外測試集的泛化誤差估計方式，OOB是隨機森林訓練過程的自然副產物、計算成本更低；交叉驗證更通用、適用於任何模型 |
| 「決策樹過度配適」 vs 「隨機森林過度配適」 | 單一決策樹的過度配適來自樹深度不受限制；隨機森林由於Bagging平均效果，較不易因樹的數量增加而過度配適，但仍可能因單棵樹深度過深而受影響 |

---

## 附錄D：本週與後續課程週次的關聯

| 後續週次 | 關聯方式 |
|---|---|
| 第 10 週：迴歸與分類 | 本週 2.4 節之三方模型比較，直接延續第10週建立的邏輯斯迴歸模型作為比較基準 |
| 第 11 週：非監督學習 | 分群結果（第11週）可作為本週樹狀模型的額外類別特徵，是監督式與非監督式學習整合應用的常見手法 |
| 第 13 週：深度學習基礎 | 樹狀模型與類神經網路是處理結構化資料的兩大典範，第13週將延伸至類神經網路能捕捉的更複雜關係型態 |
| 第 15、16 週：論文整併實戰 | 本週建立的隨機森林與SHAP解釋框架，可作為後續整合型專論中「預測模組」與「決策支援模組」的技術基礎 |

## 附錄E：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 車輛感測資料（1,000筆，線性關係） | Hour2主範例 | 決策樹過度配適示範，三方模型比較（邏輯斯迴歸表現相當或略優） |
| 彈藥庫儲存資料（300筆） | 延伸練習一 | 決策樹迴歸R²=0.16, 隨機森林R²=0.35 |
| 通信裝備故障資料（500筆） | 延伸練習二 | 隨機森林分類，AUC約0.65-0.75 |
| 含交互作用之裝備資料（2,000筆） | 論文方向一 | XGBoost調校速度約4倍於隨機森林，效能相近 |
| 車輛感測資料（1,000筆，SHAP分析） | 論文方向二 | SHAP全域重要性與吉尼重要性排序一致，個案解釋可加性驗證 |

---

## 參考資料

- 決策樹分割準則（吉尼不純度、熵、資訊增益）與隨機森林之 Bagging 集成學習原理，為機器學習領域廣泛採用之標準方法，詳細數學推導可參考 `scikit-learn` 官方文件（[Decision Trees](https://scikit-learn.org/stable/modules/tree.html)、[Ensemble Methods](https://scikit-learn.org/stable/modules/ensemble.html)）。
- SHAP 方法之理論基礎與原始論文：Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions*。

---

*下週課程：第 13 週｜深度學習基礎與類神經網路（Neural Networks Fundamentals）。*
