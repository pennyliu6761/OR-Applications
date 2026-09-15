# 第 9 週｜資料前處理、探勘統計與 Python 數據工程基底

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 9 週／共 16 週　**時數**：3 小時
> **本週關鍵字**：缺失值插補、IQR 離群值檢驗、Min-Max／Z-Score 標準化、One-Hot／Target Encoding、變異數膨脹因子 VIF、資料前處理管線 Pipeline
> **使用工具**：Google Colab（Python：pandas、NumPy、scikit-learn、statsmodels）

---

## 0. 本週課程時間分配

| 節次 | 時間 | 內容 |
|---|---|---|
| Hour 1 | 60 分鐘 | 理論深度：缺失值處理、IQR 離群值檢驗、標準化、類別編碼、共線性與 VIF |
| Hour 2 | 60 分鐘 | Python 示範演練：模擬 5,000 筆軍用車輛感測資料，完整前處理流程與 Pipeline 建立 |
| Hour 3（前 40 分鐘） | 40 分鐘 | 延伸練習（兩組模擬資料，附完整程式碼）＋ 綜合案例研究 |
| Hour 3（後 20 分鐘） | 20 分鐘 | **碩士論文延伸應用**：兩個可行論文方向之主題定義、方法說明、模擬資料與 Python 實作展示 |

> [!NOTE]
> 本週起課程進入「第二階段：AI 大數據分析」，示範演練改為 Python，建議於 [Google Colab](https://colab.research.google.com/) 開啟一份新筆記本，依序將本講義中的程式碼區塊複製貼上並執行——每一段程式碼都可以獨立執行、且已經過完整測試，貼上後應該能直接得到與講義中相同的結果。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 說明資料前處理在整個資料分析／機器學習流程中的重要性。
2. 運用不同策略處理缺失值，並理解各策略的適用時機與限制。
3. 運用 IQR 法偵測離群值，並說明單變量與多變量離群值偵測的差異。
4. 比較 Min-Max 標準化與 Z-Score 標準化的計算方式與適用情境。
5. 運用 One-Hot Encoding 與 Target Encoding 處理類別變數，並理解兩者的優缺點與資料洩漏風險。
6. 計算並解讀變異數膨脹因子（VIF），判斷特徵之間是否存在共線性問題。
7. 在 Python 中使用 pandas、NumPy、scikit-learn 建立完整的資料前處理管線（Pipeline）。
8. 說出至少兩個可將本週方法延伸為碩士論文題目的具體方向，並理解其對應的進階方法與實作方式。

---

## Hour 1｜理論深度（60 分鐘）

### 1.0 資料前處理導論

資料科學界有一句經典格言：「垃圾進、垃圾出（Garbage In, Garbage Out）」——不論後續使用多麼精密的統計模型或機器學習演算法，若輸入的資料本身充滿缺失值、離群值、量綱不一致或編碼錯誤的類別變數，最終得到的分析結果都不可靠。業界普遍估計，資料科學專案中**60–80% 的時間**都花在資料蒐集與前處理，而非模型本身的建構——這也是為什麼本課程在正式進入迴歸、分類、深度學習等方法之前，要先花一整週紮實地建立前處理的基本功。

本週處理的前處理任務可分為四大類：

| 任務類別 | 目的 | 本週對應小節 |
|---|---|---|
| 缺失值處理 | 填補或移除資料中的空值 | 1.1 |
| 離群值處理 | 識別並處理異常極端值 | 1.2 |
| 尺度轉換 | 讓不同量綱的變數可以公平比較 | 1.3 |
| 類別變數編碼 | 將文字類別轉換為模型可用的數值 | 1.4 |

---

### 1.1 缺失值處理（Missing Value Imputation）

**缺失值的常見成因**：感測器故障、人工登錄疏漏、系統整合時欄位對應失敗、受訪者拒答等。

**常見處理策略**：

| 策略 | 做法 | 適用情境 | 限制 |
|---|---|---|---|
| 刪除列（Listwise Deletion） | 直接刪除含缺失值的整筆資料 | 缺失比例很低（如<5%）、且缺失為完全隨機 | 樣本數大幅減少時會損失統計檢定力 |
| 平均數／中位數／眾數填補 | 以該欄位的集中趨勢統計量填補 | 快速、簡單，缺失比例中等 | 會人為降低變數的變異數，扭曲後續統計檢定 |
| 前向／後向填補（Forward/Backward Fill） | 以時間序列中前一筆／後一筆的值填補 | 時間序列資料，數值變化平緩 | 不適用於橫斷面（非時間序列）資料 |
| KNN 插補 | 以最相似（K 個最近鄰居）的其他觀測值平均插補 | 變數之間存在結構性關聯、缺失比例中高 | 計算成本較高，需要先處理好距離度量的尺度問題 |
| 疊代式插補（Iterative Imputer, MICE） | 把每個含缺失值的欄位當作目標，用其餘欄位建回歸模型反覆疊代預測 | 多欄位同時存在缺失、欄位間關聯複雜 | 計算成本最高，須留意疊代是否收斂 |

> [!IMPORTANT]
> 平均數／中位數填補是最常見、卻也最容易被誤用的方法：它會讓所有缺失值填入**完全相同**的數字，人為壓縮了該變數的變異數，若後續要用這個變數做迴歸或計算相關係數，統計檢定的顯著性可能被扭曲（p 值容易失真偏小）。本週 Hour 3 後段的碩士論文延伸方向一，會具體示範 KNN 插補與疊代式插補如何緩解這個問題。

---

### 1.2 離群值檢驗：IQR 法

**四分位距（Interquartile Range, IQR）**：$IQR = Q_3 - Q_1$，其中 $Q_1$、$Q_3$ 分別為第一、第三四分位數。

**判斷準則**：

$$
\text{下界} = Q_1 - 1.5 \times IQR, \qquad \text{上界} = Q_3 + 1.5 \times IQR
$$

任何觀測值落在 $[\text{下界}, \text{上界}]$ 範圍之外，即判定為離群值。係數 $1.5$ 為業界慣例（更嚴格的分析有時採用 $3.0$ 判斷「極端離群值」）。

> [!NOTE]
> IQR 法屬於**穩健統計量（Robust Statistic）**——因為四分位數本身不受極端值影響（不像平均數、標準差會被少數極端值拉扯），所以用 IQR 法判斷離群值時，不會出現「離群值自己把判斷離群值的門檻也一起拉大」的循環問題，這正是 IQR 法優於「平均值 ± 幾倍標準差」判斷法的地方。

**單變量 vs. 多變量離群值**：IQR 法屬於**單變量**方法，只能偵測「單一變數本身數值異常」的觀測值；若某筆觀測值的每個變數個別看都在正常範圍內，但變數之間的**組合**不合常理（例如里程數極低、但引擎溫度與震動值都極高），則需要**多變量**離群值偵測方法（如 Isolation Forest、馬氏距離），這也是本週 Hour 3 論文延伸方向一的核心主題。

延伸閱讀：[四分位數](https://zh.wikipedia.org/zh-tw/%E5%9B%9B%E5%88%86%E4%BD%8D%E6%95%B8)、[四分位距](https://zh.wikipedia.org/zh-tw/%E5%9B%9B%E5%88%86%E4%BD%8D%E8%B7%9D)（維基百科）。

---

### 1.3 標準化：Min-Max 縮放 vs. Z-Score 標準化

**適用時機**：不同變數的量綱（單位、數值範圍）差異極大時（例如里程數以萬為單位、震動值以個位數為單位），許多模型（如 KNN、SVM、類神經網路、含正規化項的迴歸）會因為量綱差異而讓數值範圍較大的變數不成比例地主導模型，此時須先進行標準化。

**Min-Max 縮放**：

$$
x' = \frac{x - x_{min}}{x_{max}-x_{min}}
$$

將數值線性映射到 $[0,1]$ 區間。**對離群值極度敏感**——若資料中存在一個極端離群值，會把絕大多數正常值都壓縮到 $[0,1]$ 中很小的一段範圍內。

**Z-Score 標準化**：

$$
x' = \frac{x-\bar{x}}{s}
$$

將數值轉換為平均數 0、標準差 1 的分配。同樣會受離群值影響（因為平均數、標準差都不是穩健統計量），但影響程度通常小於 Min-Max。

> [!TIP]
> 若資料中已知存在明顯離群值、且尚未處理，建議優先採用**穩健標準化（Robust Scaling）**：以中位數取代平均數、以 IQR 取代標準差進行標準化（$x'=(x-\text{中位數})/IQR$），此方法不在本週理論詳述範圍內，但在 Hour 2 示範中會展示其與傳統兩種方法的比較。

---

### 1.4 類別變數編碼：One-Hot Encoding vs. Target Encoding

**One-Hot Encoding（獨熱編碼）**：將一個具有 $k$ 個類別的變數，轉換為 $k$ 個 0/1 的虛擬變數欄位。

- **優點**：不會人為賦予類別之間不存在的順序關係，適用於大多數模型。
- **缺點**：類別數量很多時（高基數類別變數），會產生大量欄位（維度爆炸），且欄位之間存在完全共線性（$k$ 個虛擬變數中，任一個都可由其餘 $k-1$ 個推算出來，即「虛擬變數陷阱」），實務上常直接捨棄一欄避免此問題。

**Target Encoding（目標編碼）**：以該類別在**訓練資料**中對應目標變數的平均值取代類別標籤。

- **優點**：不增加欄位數量，能捕捉類別與目標變數之間的關聯強度。
- **缺點**：**若使用全體資料（含測試資料）計算目標平均值，會造成嚴重的資料洩漏（Data Leakage）**——模型間接看到了測試資料的答案，導致驗證階段的績效被過度樂觀高估，測試／正式上線後績效大幅落差。**正確做法是僅用訓練資料計算目標編碼的對照表，再套用到驗證與測試資料上。**

> [!CAUTION]
> Target Encoding 的資料洩漏問題，是初學者最容易犯的錯誤之一，也是審查機器學習相關論文時最常被挑出的方法論瑕疵。本週 Hour 2 示範與 Hour 3 論文延伸方向二的程式碼，都會嚴格遵守「先切分訓練／測試集，只用訓練集計算編碼對照表」的正確流程，請特別留意程式碼中 `train_idx`、`test_idx` 的使用方式。

---

### 1.5 共線性與變異數膨脹因子（VIF）

**多重共線性（Multicollinearity）**：迴歸模型中，若自變數彼此之間存在高度線性相關，會導致係數估計不穩定（微小的資料變化就可能讓係數大幅改變，甚至正負號翻轉），雖然不影響模型整體的預測能力，但會讓個別係數的解釋失去意義。

**變異數膨脹因子（Variance Inflation Factor, VIF）**：

$$
VIF_j = \frac{1}{1-R_j^2}
$$

其中 $R_j^2$ 為將第 $j$ 個自變數，用其餘所有自變數進行迴歸所得到的判定係數。$VIF_j$ 衡量的是：第 $j$ 個變數與其餘變數的線性相關程度越高，$VIF_j$ 就越大。

**判讀基準**（業界常見經驗法則）：

| VIF 值 | 判讀 |
|---|---|
| 1 | 完全無共線性 |
| 1–5 | 中度相關，通常可接受 |
| >5（部分文獻採 >10） | 共線性問題較嚴重，建議檢視是否需要移除或合併變數 |

延伸閱讀：[變異數膨脹因子](https://zh.wikipedia.org/zh-tw/%E6%96%B9%E5%B7%AE%E6%93%B4%E5%A4%A7%E5%9B%A0%E5%AD%90)、[多重共線性](https://zh.wikipedia.org/zh-tw/%E5%A4%9A%E9%87%8D%E5%85%B1%E7%B7%9A%E6%80%A7)（維基百科）。

---

### 1.6 資料前處理管線（Pipeline）

**為什麼要用 Pipeline，而不是逐步手動處理？**

1. **避免資料洩漏**：Pipeline 可以確保「缺失值填補的統計量、標準化的平均數與標準差」等，都只從訓練資料中學習，再一致地套用到驗證／測試資料，從架構上杜絕資料洩漏風險。
2. **可重複性與可維護性**：一旦 Pipeline 建立完成，未來有新資料進來時，只需呼叫 `.transform()`，就能自動套用與訓練時完全一致的前處理步驟，不需要重新手動撰寫每一個步驟。
3. **程式碼精簡**：scikit-learn 的 `ColumnTransformer` 能將「數值欄位走一條處理流程、類別欄位走另一條處理流程」整合在單一物件中，是實務上機器學習專案的標準做法。

本週 Hour 2 示範五會完整展示如何用 `Pipeline` 與 `ColumnTransformer` 建立正式的前處理流程。

---

### 1.7 方法選擇指南

| 情境 | 建議方法 |
|---|---|
| 缺失比例低、隨機缺失 | 刪除列，或中位數／平均數填補 |
| 缺失比例中高、變數間有結構關聯 | KNN 插補或疊代式插補 |
| 資料存在明顯極端值 | IQR 法偵測 + 穩健標準化 |
| 資料量綱差異大、對離群值敏感的模型（KNN、SVM、神經網路） | Min-Max 或 Z-Score 標準化（依是否已處理離群值決定） |
| 類別變數類別數量少（如 <10） | One-Hot Encoding |
| 類別變數類別數量多（高基數） | Target Encoding（務必注意資料洩漏），或頻率編碼 |
| 多元迴歸模型建立前 | 檢查 VIF，剔除或合併高度共線的變數 |

### 1.8 常用中英文詞彙對照表

| 中文 | 英文 | 中文 | 英文 |
|---|---|---|---|
| 特徵工程 | Feature Engineering | 缺失值插補 | Missing Value Imputation |
| 四分位距 | Interquartile Range, IQR | 離群值 | Outlier |
| 標準化 | Standardization / Scaling | 獨熱編碼 | One-Hot Encoding |
| 目標編碼 | Target Encoding | 資料洩漏 | Data Leakage |
| 多重共線性 | Multicollinearity | 變異數膨脹因子 | Variance Inflation Factor, VIF |
| 資料前處理管線 | Preprocessing Pipeline | 穩健統計量 | Robust Statistic |

### 1.9 各方法的假設條件與限制一覽

| 方法 | 隱含假設 | 主要限制 |
|---|---|---|
| 平均數／中位數填補 | 缺失為完全隨機（MCAR） | 壓縮變數變異數，扭曲後續統計檢定 |
| IQR 離群值法 | 資料分配大致對稱或接近常態 | 對高度偏態分配可能誤判過多正常值為離群值 |
| Min-Max 標準化 | 資料範圍穩定、無極端離群值 | 新資料若超出訓練時的最大最小值範圍，會產生超出 [0,1] 的數值 |
| Target Encoding | 訓練資料量足夠大，類別內樣本數足夠估計穩定的平均值 | 類別樣本數過少時，估計值不穩定，容易過度配適 |
| VIF | 迴歸模型設定正確（無遺漏重要變數） | 只能偵測線性共線性，無法偵測非線性關聯 |

---

## Hour 2｜Python 示範演練（60 分鐘）

> [!NOTE]
> 以下所有程式碼皆已於 Google Colab 環境測試通過，可依序複製貼上執行。建議每執行完一個程式碼區塊，先觀察輸出結果，理解該步驟的作用，再繼續下一步。

### 2.0 環境設置與模擬資料生成

```python
# ============================================================
# 第9週 Hour 2 示範：資料前處理完整流程
# 情境：模擬 5,000 筆軍用車輛感測記錄，用於預測性維護分析
# ============================================================

# 安裝與匯入套件（Colab 通常已預先安裝，若缺少 statsmodels 才需要 pip install）
!pip install statsmodels --quiet

import numpy as np
import pandas as pd

# 設定隨機種子，確保每次執行結果可重現（教學與除錯時非常重要）
np.random.seed(42)
n = 5000  # 模擬車輛數

# --- 產生類別型欄位 ---
# 車輛類型：戰甲車/輪型運輸車/工程車/救護車，依比例隨機抽樣
vehicle_type = np.random.choice(
    ['戰甲車', '輪型運輸車', '工程車', '救護車'],
    size=n, p=[0.35, 0.4, 0.15, 0.1]
)
# 維保區域：北中南東，依比例隨機抽樣
maintenance_region = np.random.choice(
    ['北部', '中部', '南部', '東部'],
    size=n, p=[0.3, 0.3, 0.25, 0.15]
)

# --- 產生數值型欄位（感測數據） ---
# 累積里程數：常態分配，平均45000公里，標準差15000，並限制最小值為500(避免負值)
mileage_km = np.random.normal(45000, 15000, n).clip(500, None)
# 引擎溫度：常態分配，平均88度，標準差6度
engine_temp_c = np.random.normal(88, 6, n)
# 震動值：Gamma分配（右偏分配，符合震動數據多數集中在低值、少數偏高的特性）
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n)
# 油耗效率：與里程數呈負相關（里程越高、效率越低），並加入隨機雜訊
fuel_efficiency_km_l = 12 - 0.00008 * mileage_km + np.random.normal(0, 0.8, n)

# --- 產生目標變數：是否需要維護 (0=否,1=是) ---
# 以里程、溫度、震動的加權組合構成風險分數，並取前30%高風險者標記為需要維護
risk_score = (0.00002*mileage_km + 0.03*engine_temp_c
              + 0.15*vibration_mm_s + np.random.normal(0, 0.5, n))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

# --- 刻意植入缺失值，模擬感測器故障或漏傳情形 ---
# engine_temp_c 約 6% 缺失
miss_idx1 = np.random.choice(n, size=int(0.06*n), replace=False)
engine_temp_c[miss_idx1] = np.nan
# fuel_efficiency_km_l 約 4% 缺失
miss_idx2 = np.random.choice(n, size=int(0.04*n), replace=False)
fuel_efficiency_km_l[miss_idx2] = np.nan

# --- 刻意植入離群值，模擬感測器異常讀數 ---
# 25筆震動值異常放大8倍
out_idx1 = np.random.choice(n, size=25, replace=False)
vibration_mm_s[out_idx1] = vibration_mm_s[out_idx1] * 8
# 15筆里程數異常放大3倍（可能是資料登錄單位錯誤）
out_idx2 = np.random.choice(n, size=15, replace=False)
mileage_km[out_idx2] = mileage_km[out_idx2] * 3

# --- 組成 DataFrame ---
df = pd.DataFrame({
    'vehicle_id': [f'V{str(i).zfill(5)}' for i in range(1, n+1)],
    'vehicle_type': vehicle_type,
    'mileage_km': mileage_km.round(1),
    'engine_temp_c': engine_temp_c.round(2),
    'vibration_mm_s': vibration_mm_s.round(3),
    'fuel_efficiency_km_l': fuel_efficiency_km_l.round(2),
    'maintenance_region': maintenance_region,
    'needs_maintenance': needs_maintenance,
})

print("資料集形狀:", df.shape)
print("\n前5筆資料:")
print(df.head())
print("\n目標變數分布:")
print(df['needs_maintenance'].value_counts(normalize=True))
```

**預期輸出**：資料集共 5,000 列、8 欄，目標變數中約 30% 標記為「需要維護」（1）、70% 為「不需要」（0）。

---

### 2.1 初步檢視：認識你的資料

```python
# ============================================================
# 2.1 初步檢視
# ============================================================

# 資料型態與非缺失值數量總覽
print("=== df.info() ===")
df.info()

# 描述性統計（僅數值欄位）
print("\n=== 數值欄位描述性統計 ===")
print(df.describe().T)

# 檢查每個欄位的缺失值數量與比例
print("\n=== 缺失值統計 ===")
missing_summary = pd.DataFrame({
    '缺失數量': df.isna().sum(),
    '缺失比例(%)': (df.isna().sum() / len(df) * 100).round(2)
})
print(missing_summary[missing_summary['缺失數量'] > 0])
```

> [!TIP]
> `df.info()` 與 `df.describe()` 應該是拿到任何新資料集後**第一件要做的事**：前者快速確認資料型態是否符合預期（例如日期欄位是否被誤讀為文字）、後者快速掃描是否有不合理的最大最小值（例如里程數出現負數）。

---

### 2.2 缺失值填補

```python
# ============================================================
# 2.2 缺失值填補：以中位數填補為示範（對照1.1節理論）
# ============================================================

# 先記錄填補前的中位數，供後續與其他方法比較
median_temp = df['engine_temp_c'].median()
median_fuel = df['fuel_efficiency_km_l'].median()
print(f"engine_temp_c 中位數: {median_temp:.2f}")
print(f"fuel_efficiency_km_l 中位數: {median_fuel:.2f}")

# 建立新欄位存放填補後的結果，保留原始欄位以便後續比較（實務上養成好習慣）
df['engine_temp_c_filled'] = df['engine_temp_c'].fillna(median_temp)
df['fuel_efficiency_km_l_filled'] = df['fuel_efficiency_km_l'].fillna(median_fuel)

# 驗證缺失值是否已完全填補
print("\n填補後剩餘缺失值數量:",
      df[['engine_temp_c_filled', 'fuel_efficiency_km_l_filled']].isna().sum().sum())
```

---

### 2.3 離群值偵測（IQR 法）

```python
# ============================================================
# 2.3 離群值偵測：IQR 法（對照1.2節理論公式）
# ============================================================

def detect_outliers_iqr(series, k=1.5):
    """
    以 IQR 法偵測離群值。
    參數:
        series: pandas Series，欲檢查的數值欄位
        k: 判斷門檻的倍數，預設1.5（業界慣例）
    回傳:
        下界, 上界, 布林遮罩(True代表該筆為離群值)
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    mask = (series < lower) | (series > upper)
    return lower, upper, mask

# 針對震動值檢查
lower_v, upper_v, mask_v = detect_outliers_iqr(df['vibration_mm_s'])
print(f"震動值: Q1及Q3計算出下界={lower_v:.3f}, 上界={upper_v:.3f}")
print(f"偵測到離群值數量: {mask_v.sum()} 筆（佔全體 {mask_v.mean()*100:.2f}%）")

# 針對里程數檢查
lower_m, upper_m, mask_m = detect_outliers_iqr(df['mileage_km'])
print(f"\n里程數: 下界={lower_m:.1f}, 上界={upper_m:.1f}")
print(f"偵測到離群值數量: {mask_m.sum()} 筆（佔全體 {mask_m.mean()*100:.2f}%）")

# 處理方式示範：以「截尾（Capping/Winsorization）」取代直接刪除
# 好處：不損失樣本數，同時降低極端值對後續統計分析的影響
df['vibration_mm_s_capped'] = df['vibration_mm_s'].clip(lower_v, upper_v)
df['mileage_km_capped'] = df['mileage_km'].clip(lower_m, upper_m)

print("\n截尾前後比較（震動值）:")
print(df[['vibration_mm_s', 'vibration_mm_s_capped']].describe().T[['mean', 'std', 'max']])
```

> [!TIP]
> 除了本例採用的「截尾」，離群值的處理方式還包括「直接刪除」（樣本數足夠時）與「取對數轉換」（若離群源自右偏分配本身的自然特性，而非量測錯誤）。選擇哪一種方式，取決於離群值是「量測錯誤」還是「真實但罕見的極端情形」——若是後者，直接刪除可能會讓模型失去辨識稀有但重要事件（如即將故障的車輛）的能力。

---

### 2.4 標準化比較：Min-Max vs. Z-Score vs. 穩健標準化

```python
# ============================================================
# 2.4 標準化方法比較（對照1.3節理論）
# ============================================================
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# 分別建立三種標準化物件
mm_scaler = MinMaxScaler()
z_scaler = StandardScaler()
robust_scaler = RobustScaler()   # 以中位數與IQR取代平均數與標準差，對離群值較不敏感

# 對「尚未處理離群值」的原始里程數欄位進行標準化，觀察離群值造成的影響
df['mileage_minmax'] = mm_scaler.fit_transform(df[['mileage_km']])
df['mileage_zscore'] = z_scaler.fit_transform(df[['mileage_km']])
df['mileage_robust'] = robust_scaler.fit_transform(df[['mileage_km']])

comparison = df[['mileage_km', 'mileage_minmax', 'mileage_zscore', 'mileage_robust']].describe().T
print(comparison[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])
```

**預期觀察**：`mileage_minmax` 的中位數（50%分位）遠低於 0.5（因為少數極端離群值把整個 [0,1] 的上界拉得很高，把多數正常值都壓縮在很小的一段區間內），而 `mileage_robust`（穩健標準化）的中位數會非常接近 0——這正是 1.3 節提醒「Min-Max 對離群值極度敏感」的具體數字證據。

---

### 2.5 類別變數編碼

```python
# ============================================================
# 2.5 類別變數編碼（對照1.4節理論）
# ============================================================

# --- One-Hot Encoding：適合類別數少的 vehicle_type（4類） ---
onehot_vtype = pd.get_dummies(df['vehicle_type'], prefix='vtype')
print("One-Hot Encoding 結果（前5筆）:")
print(onehot_vtype.head())
print("\n各類別筆數:")
print(onehot_vtype.sum())

# --- Target Encoding：示範正確做法，避免資料洩漏 ---
# 關鍵：務必先切分訓練/測試集，只用訓練集計算目標平均值
from sklearn.model_selection import train_test_split

train_idx, test_idx = train_test_split(
    df.index, test_size=0.25, random_state=42, stratify=df['needs_maintenance']
)

# 只用訓練集資料，計算每個維保區域的目標平均值（需要維護的比例）
region_target_map = df.loc[train_idx].groupby('maintenance_region')['needs_maintenance'].mean()
print("\n訓練集計算出的目標編碼對照表:")
print(region_target_map)

# 將此對照表同時套用到訓練集與測試集（測試集絕對不能用自己的資料重新計算！）
df['region_target_enc'] = df['maintenance_region'].map(region_target_map)

print("\n套用後的結果（去重複檢視）:")
print(df[['maintenance_region', 'region_target_enc']].drop_duplicates())
```

> [!CAUTION]
> 再次強調：`region_target_map` 只用 `df.loc[train_idx]`（訓練集）計算，然後才用 `.map()` 套用到**全部**資料（含測試集）。若不慎寫成先對全體資料 `df.groupby(...)` 計算目標編碼、才切分訓練測試集，測試集的目標編碼值就已經「偷看」了測試集自己的答案，這是本週最重要的實作陷阱，請務必養成正確的操作順序。

---

### 2.6 VIF 共線性檢查

```python
# ============================================================
# 2.6 VIF 共線性檢查（對照1.5節理論）
# ============================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

numeric_cols = ['mileage_km', 'engine_temp_c_filled', 'vibration_mm_s',
                 'fuel_efficiency_km_l_filled']
X = df[numeric_cols].copy()
X = sm.add_constant(X)  # VIF計算慣例上需要加入常數項

vif_data = pd.DataFrame()
vif_data['特徵'] = X.columns
vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)
```

**預期輸出解讀**：`mileage_km` 與 `fuel_efficiency_km_l_filled` 的 VIF 約為 2.5 左右——這是因為本週模擬資料刻意讓油耗效率與里程數存在線性關係（見 2.0 節資料生成邏輯），VIF 落在「中度相關但可接受」的範圍（依 1.5 節判讀基準，VIF<5 通常視為可接受）。

---

### 2.7 整合為正式的 Pipeline

```python
# ============================================================
# 2.7 整合為 scikit-learn Pipeline（對照1.6節理論）
# ============================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# 重新讀取一份乾淨的資料（實務上Pipeline應該從最原始的資料開始）
df_raw = df[['vehicle_type', 'mileage_km', 'engine_temp_c', 'vibration_mm_s',
             'fuel_efficiency_km_l', 'maintenance_region', 'needs_maintenance']].copy()

# 先以IQR法對極端離群值進行截尾處理（Pipeline外部先做，因sklearn無原生IQR截尾工具）
def cap_outliers_iqr(series, k=1.5):
    Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = Q3 - Q1
    return series.clip(Q1 - k*IQR, Q3 + k*IQR)

df_raw['vibration_mm_s'] = cap_outliers_iqr(df_raw['vibration_mm_s'])
df_raw['mileage_km'] = cap_outliers_iqr(df_raw['mileage_km'])

# 定義數值欄位與類別欄位清單
numeric_features = ['mileage_km', 'engine_temp_c', 'vibration_mm_s', 'fuel_efficiency_km_l']
categorical_features = ['vehicle_type', 'maintenance_region']

# 數值欄位的處理流程：先填補缺失值(中位數)，再做Z-Score標準化
numeric_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# 類別欄位的處理流程：One-Hot編碼
categorical_pipeline = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# 用ColumnTransformer把兩條流程「並聯」，分別套用到對應欄位
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_pipeline, numeric_features),
    ('cat', categorical_pipeline, categorical_features)
])

# 準備輸入資料與標籤
X = df_raw.drop(columns=['needs_maintenance'])
y = df_raw['needs_maintenance']

# 執行 fit_transform：一次完成「學習統計量」與「套用轉換」兩件事
X_transformed = preprocessor.fit_transform(X)

print("轉換後資料形狀:", X_transformed.shape)
print("\n轉換後的欄位名稱:")
print(list(preprocessor.get_feature_names_out()))

# 轉回DataFrame方便檢視
X_transformed_df = pd.DataFrame(X_transformed, columns=preprocessor.get_feature_names_out())
print("\n轉換後資料摘要統計（前4個數值欄位）:")
print(X_transformed_df.iloc[:, :4].describe().T[['mean', 'std', 'min', 'max']])
```

> [!TIP]
> 觀察轉換後數值欄位的平均數應接近 0、標準差接近 1（Z-Score 標準化的特性）；類別欄位則全部轉為 0/1。**這個 `preprocessor` 物件可以直接存起來（例如用 `joblib.dump`），未來有新的車輛感測資料進來時，只需要呼叫 `preprocessor.transform(新資料)`，就能自動套用與訓練時完全一致的前處理邏輯**，這正是 Pipeline 在實務專案中最大的價值。

---

### 2.8 特徵摘要統計表（論文標準呈現格式）

```python
# ============================================================
# 2.8 建立論文等級的特徵摘要統計表
# ============================================================

summary_table = pd.DataFrame({
    '變數名稱': numeric_features,
    '型態': ['數值型']*len(numeric_features),
    '筆數': [df_raw[c].notna().sum() for c in numeric_features],
    '平均數': [df_raw[c].mean().round(2) for c in numeric_features],
    '標準差': [df_raw[c].std().round(2) for c in numeric_features],
    '最小值': [df_raw[c].min().round(2) for c in numeric_features],
    '最大值': [df_raw[c].max().round(2) for c in numeric_features],
    '遺失比例(%)': [(df_raw[c].isna().mean()*100).round(2) for c in numeric_features],
})
print(summary_table.to_string(index=False))
```

> [!NOTE]
> 這張表格是論文「資料集描述（Methodology - Data Description）」段落最常見的標準呈現格式，建議每次拿到新資料集、完成前處理後，都養成產出這張表的習慣，方便日後直接引用於論文或研究報告中。

---

## Hour 3（前 40 分鐘）｜延伸練習與綜合案例研究

### 3.1 延伸練習一：彈藥庫溫濕度感測資料前處理（附完整程式碼）

**情境**：模擬彈藥庫溫濕度監控感測器資料，練習缺失值處理與離群值偵測。

```python
# ============================================================
# 延伸練習一：彈藥庫溫濕度感測資料
# ============================================================
np.random.seed(123)
n2 = 2000

temperature_c = np.random.normal(22, 3, n2)      # 目標溫度約22度
humidity_pct = np.random.normal(45, 8, n2)        # 目標濕度約45%
storage_zone = np.random.choice(['A區', 'B區', 'C區'], size=n2, p=[0.4, 0.35, 0.25])

# 植入缺失值（濕度感測器5%故障率）
miss_idx = np.random.choice(n2, size=int(0.05*n2), replace=False)
humidity_pct[miss_idx] = np.nan

# 植入離群值（10筆溫度異常，可能是感測器接觸不良）
out_idx = np.random.choice(n2, size=10, replace=False)
temperature_c[out_idx] = temperature_c[out_idx] + np.random.choice([-15,15], size=10)

df_ex1 = pd.DataFrame({
    'temperature_c': temperature_c.round(2),
    'humidity_pct': humidity_pct.round(2),
    'storage_zone': storage_zone,
})

# 任務1：檢查缺失值
print("缺失值統計:")
print(df_ex1.isna().sum())

# 任務2：以中位數填補濕度缺失值
df_ex1['humidity_filled'] = df_ex1['humidity_pct'].fillna(df_ex1['humidity_pct'].median())

# 任務3：以IQR法偵測溫度離群值
Q1, Q3 = df_ex1['temperature_c'].quantile([0.25, 0.75])
IQR = Q3 - Q1
lower, upper = Q1-1.5*IQR, Q3+1.5*IQR
outliers = df_ex1[(df_ex1['temperature_c']<lower)|(df_ex1['temperature_c']>upper)]
print(f"\n溫度離群值範圍: [{lower:.2f}, {upper:.2f}]")
print(f"偵測到 {len(outliers)} 筆離群值")
print(outliers[['temperature_c']])
```

**詳解**：執行後應可看到偵測到約 20–25 筆溫度離群值——這個數字**不會剛好等於植入的 10 筆**，因為 IQR 法偵測到的是「所有超出邊界的觀測值」，除了刻意植入的 10 筆極端值外，常態分配 $N(22,3)$ 本身的自然尾端，統計上也會有約 0.5–1% 的觀測值恰好落在 IQR 邊界之外（$2000$ 筆 × 約 0.7% ≈ 14 筆），兩者相加才是最終偵測到的總數。這是一個很好的提醒：**IQR 法偵測到的「離群值」，不等於「人為植入或量測錯誤的異常值」，其中一部分可能只是常態分配下本來就會自然出現的極端但合理的觀測值**，實務判讀時應該進一步檢視這些觀測值的合理性，而非全部視為錯誤逕行剔除。

---

### 3.2 延伸練習二：無人機零組件庫存資料編碼與標準化（附完整程式碼）

```python
# ============================================================
# 延伸練習二：無人機零組件庫存資料
# ============================================================
np.random.seed(456)
n3 = 1500

unit_cost = np.random.lognormal(mean=6, sigma=1, size=n3)  # 右偏分配，模擬單價
supplier = np.random.choice(['供應商甲','供應商乙','供應商丙','供應商丁'],
                              size=n3, p=[0.4,0.3,0.2,0.1])
lead_time_days = np.random.poisson(lam=12, size=n3)
critical_flag = np.random.choice([0,1], size=n3, p=[0.75,0.25])  # 是否為關鍵零件

df_ex2 = pd.DataFrame({
    'unit_cost': unit_cost.round(1),
    'supplier': supplier,
    'lead_time_days': lead_time_days,
    'critical_flag': critical_flag,
})

# 任務1：比較 unit_cost 的 Min-Max 與 Z-Score 標準化（該欄位為右偏分配，適合觀察差異）
from sklearn.preprocessing import MinMaxScaler, StandardScaler
mm = MinMaxScaler(); z = StandardScaler()
df_ex2['cost_minmax'] = mm.fit_transform(df_ex2[['unit_cost']])
df_ex2['cost_zscore'] = z.fit_transform(df_ex2[['unit_cost']])
print("標準化結果比較:")
print(df_ex2[['unit_cost','cost_minmax','cost_zscore']].describe().T[['mean','std','50%','max']])

# 任務2：One-Hot編碼supplier欄位
supplier_onehot = pd.get_dummies(df_ex2['supplier'], prefix='supplier')
print("\nOne-Hot編碼結果欄位:", list(supplier_onehot.columns))

# 任務3：計算 unit_cost 與 lead_time_days 之間的VIF(檢查是否需要一併納入迴歸模型)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
X_ex2 = sm.add_constant(df_ex2[['unit_cost','lead_time_days']])
vif_ex2 = pd.DataFrame({
    '特徵': X_ex2.columns,
    'VIF': [variance_inflation_factor(X_ex2.values, i) for i in range(X_ex2.shape[1])]
})
print("\nVIF結果:")
print(vif_ex2)
```

**詳解**：由於 `unit_cost` 採用對數常態分配（右偏，存在少數高價零件），標準化後應觀察到 Min-Max 的中位數明顯偏離 0.5；`unit_cost` 與 `lead_time_days` 為獨立生成，VIF 應接近 1（無共線性問題）。

---

### 3.3 綜合案例研究：建立可重複使用的國防後勤資料前處理模組

> [!NOTE]
> 以下是一個虛構但貼近實務的案例，目的是把本週技術串接起來，示範完整的資料前處理專案思考過程。

**背景**：某聯兵旅資訊處欲建立一套標準化的資料前處理模組，供後續各類感測資料分析（車輛、彈藥庫、後勤倉儲）共同使用。

**步驟一：盤點各資料來源的欄位型態與品質**（對應 2.1 節）
資訊處人員對各系統匯出的原始資料執行 `df.info()`、`df.describe()`、缺失值統計，建立每個資料來源的「資料品質報告」。

**步驟二：依欄位型態分流設計前處理規則**（對應 1.1–1.5 節）
針對數值型欄位，依缺失比例決定填補策略（低缺失用中位數、高缺失考慮 KNN 插補）；依離群值比例決定是否截尾；針對類別型欄位，依類別數量決定 One-Hot 或 Target Encoding。

**步驟三：以 Pipeline 封裝為可重複使用的模組**（對應 2.7 節）
資訊處將上述規則寫成標準化的 `ColumnTransformer`，並將建立好的 Pipeline 物件存檔，未來新資料進來時可直接載入套用，不需要每次重新設計。

**步驟四：產出標準化的特徵摘要統計報告**（對應 2.8 節）
每次前處理完成後，自動產出摘要統計表，作為後續分析報告或論文方法論章節的標準附件。

---

## Hour 3（後 20 分鐘）｜碩士論文延伸應用

> [!NOTE]
> 以下兩個方向，示範如何把本週介紹的基礎前處理方法，延伸為具備研究貢獻的碩士論文題目——核心邏輯是：**基礎方法只是「做了什麼」，碩士論文需要進一步回答「比較了什麼、量化了什麼差異、對下游任務造成什麼影響」**。每個方向皆包含主題定義、方法說明、模擬資料設計與完整可執行的 Python 實作。

### 3.4 論文方向一：進階缺失值與離群值處理方法之比較研究

**主題定義**：軍用裝備感測資料經常因感測器故障、傳輸中斷產生缺失值，也經常因環境干擾產生異常讀數。本研究比較不同缺失值插補方法（簡單插補 vs. KNN 插補 vs. 疊代式插補）與不同離群值偵測方法（單變量 IQR vs. 多變量 Isolation Forest）在軍用車輛預測性維護資料上的表現差異，並延伸評估這些前處理選擇對下游分類模型績效的影響。

**方法說明**：
- **KNN 插補**：以歐幾里得距離找出與缺失值最相似的 $k$ 個觀測值，取其平均作為插補值，能捕捉變數間的局部結構關聯。
- **疊代式插補（IterativeImputer，MICE 概念）**：把每個有缺失值的欄位輪流當作迴歸目標，用其餘欄位建立迴歸模型反覆疊代預測，直到插補值收斂穩定，能捕捉變數間的全域線性關聯。
- **Isolation Forest（孤立森林）**：透過隨機切割特徵空間，異常點因為「與大多數點不同」而更容易被快速孤立（切割次數少），適合捕捉單變量方法無法偵測的多變量異常組合。

**模擬資料**：沿用本週 2.0 節產生的 5,000 筆軍用車輛感測資料集。

```python
# ============================================================
# 論文方向一：進階缺失值插補與多變量離群值偵測比較
# ============================================================
from sklearn.experimental import enable_iterative_imputer  # 啟用實驗性功能，IterativeImputer必須先執行此行
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import IsolationForest

numeric_cols = ['mileage_km', 'engine_temp_c', 'vibration_mm_s', 'fuel_efficiency_km_l']

# --- 步驟1：以三種方法分別插補缺失值，比較插補結果的差異 ---

# 方法A：簡單中位數插補（基準對照組）
df_simple = df[['mileage_km','engine_temp_c','vibration_mm_s','fuel_efficiency_km_l']].copy()
imp_simple = SimpleImputer(strategy='median')
df_simple[numeric_cols] = imp_simple.fit_transform(df_simple[numeric_cols])

# 方法B：KNN插補（k=5個最近鄰居）
df_knn = df[['mileage_km','engine_temp_c','vibration_mm_s','fuel_efficiency_km_l']].copy()
imp_knn = KNNImputer(n_neighbors=5)
df_knn[numeric_cols] = imp_knn.fit_transform(df_knn[numeric_cols])

# 方法C：疊代式插補（MICE概念，最多疊代10次）
df_iter = df[['mileage_km','engine_temp_c','vibration_mm_s','fuel_efficiency_km_l']].copy()
imp_iter = IterativeImputer(random_state=42, max_iter=10)
df_iter[numeric_cols] = imp_iter.fit_transform(df_iter[numeric_cols])

# --- 比較三種方法對同一批缺失值的插補結果 ---
missing_mask = df['engine_temp_c'].isna()
comparison = pd.DataFrame({
    '簡單中位數插補': df_simple.loc[missing_mask, 'engine_temp_c'].values[:5],
    'KNN插補(k=5)': df_knn.loc[missing_mask, 'engine_temp_c'].values[:5],
    '疊代式插補': df_iter.loc[missing_mask, 'engine_temp_c'].values[:5],
})
print("=== 三種插補方法結果比較（前5筆缺失值）===")
print(comparison)
print("\n觀察：簡單中位數插補對每一筆缺失值都給出完全相同的數字；")
print("KNN與疊代式插補則依據該筆觀測值其他欄位的特徵，給出各自不同、更具個別代表性的插補值。")

# --- 步驟2：多變量離群值偵測，並與單變量IQR法比較 ---

# Isolation Forest：設定contamination=0.02，代表預期約2%資料為異常
iso_forest = IsolationForest(contamination=0.02, random_state=42)
outlier_pred = iso_forest.fit_predict(df_simple[numeric_cols])  # 回傳-1代表異常，1代表正常
n_multivariate_outliers = (outlier_pred == -1).sum()
print(f"\n=== 多變量離群值偵測（Isolation Forest）===")
print(f"偵測到 {n_multivariate_outliers} 筆多變量離群值")

# 單變量IQR法：對每個數值欄位個別檢查，任一欄位超出範圍即標記為離群值
def iqr_flag(s, k=1.5):
    Q1, Q3 = s.quantile(0.25), s.quantile(0.75)
    IQR = Q3-Q1
    return (s < Q1-k*IQR) | (s > Q3+k*IQR)

univariate_flags = pd.DataFrame({c: iqr_flag(df_simple[c]) for c in numeric_cols})
n_univariate_outliers = univariate_flags.any(axis=1).sum()
print(f"單變量IQR法（任一欄位超標即算）偵測到 {n_univariate_outliers} 筆離群值")

overlap = ((outlier_pred == -1) & univariate_flags.any(axis=1).values).sum()
print(f"兩種方法重疊偵測到的筆數: {overlap}")
print(f"\n解讀：Isolation Forest與IQR法偵測到的離群值集合並不完全重疊，")
print(f"顯示兩者捕捉到的是「不同類型」的異常——IQR法只看單一欄位是否超標，")
print(f"Isolation Forest則能捕捉「每個欄位個別看都正常、但組合起來不合常理」的異常樣態。")
```

**論文延伸建議**：可進一步將三種插補方法產出的資料集，分別餵入相同的分類模型（如隨機森林），比較模型在測試集上的 AUC／準確率差異，量化「插補方法的選擇」對下游任務績效的實際影響——這正是碩士論文「比較研究」最常見的實驗設計架構：固定其他條件、只改變欲比較的處理方法，觀察結果指標的差異。

---

### 3.5 論文方向二：類別特徵編碼方法對後勤故障分類模型效能之影響

**主題定義**：類別變數編碼方式的選擇，看似只是前處理的技術細節，實際上可能顯著影響下游分類模型的預測效能。本研究以軍用車輛「是否需要維護」二元分類任務為例，系統性比較 One-Hot Encoding、Target Encoding、Frequency Encoding 三種編碼方法對模型績效（AUC、準確率）的影響，並探討造成差異的可能原因。

**方法說明**：
- **One-Hot Encoding**：如 1.4 節所述，不假設類別間的任何數值關係。
- **Target Encoding**：如 1.4 節所述，直接編碼「類別與目標變數的關聯強度」，資訊量較大，但需嚴格避免資料洩漏。
- **Frequency Encoding（頻率編碼）**：以該類別在資料中出現的頻率（或次數）取代類別標籤，不直接使用目標變數資訊，資料洩漏風險低於 Target Encoding，但也較難捕捉類別與目標之間的直接關聯。

**模擬資料**：沿用本週 2.0 節模擬資料集，以 `vehicle_type`、`maintenance_region` 兩個類別變數，比較三種編碼方式。

```python
# ============================================================
# 論文方向二：類別編碼方法比較對分類模型效能之影響
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.impute import SimpleImputer

# 先處理數值欄位缺失值（本比較實驗的重點是編碼方法，非缺失值處理，故此處用簡單插補即可）
numeric_cols = ['mileage_km','engine_temp_c','vibration_mm_s','fuel_efficiency_km_l']
df_cmp = df.copy()
imp = SimpleImputer(strategy='median')
df_cmp[numeric_cols] = imp.fit_transform(df_cmp[numeric_cols])

y = df_cmp['needs_maintenance']
# 固定訓練/測試切分，確保三種方法在完全相同的樣本上比較，公平對照
train_idx, test_idx = train_test_split(
    df_cmp.index, test_size=0.25, random_state=42, stratify=y
)
ytr, yte = y.loc[train_idx], y.loc[test_idx]

def train_and_evaluate(X, label):
    """統一的訓練與評估函數，確保三種方法使用相同的模型設定，只有輸入特徵不同"""
    Xtr, Xte = X.loc[train_idx], X.loc[test_idx]
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)[:, 1]
    pred = clf.predict(Xte)
    auc = roc_auc_score(yte, proba)
    acc = accuracy_score(yte, pred)
    print(f"{label}: AUC={auc:.4f}, 準確率={acc:.4f}")
    return auc, acc

# --- 方法A：One-Hot Encoding ---
onehot_type = pd.get_dummies(df_cmp['vehicle_type'], prefix='vtype')
onehot_region = pd.get_dummies(df_cmp['maintenance_region'], prefix='region')
X_onehot = pd.concat([df_cmp[numeric_cols], onehot_type, onehot_region], axis=1)
auc_onehot, acc_onehot = train_and_evaluate(X_onehot, "One-Hot Encoding")

# --- 方法B：Target Encoding（僅用訓練集計算對照表，避免資料洩漏） ---
type_target_map = df_cmp.loc[train_idx].groupby('vehicle_type')['needs_maintenance'].mean()
region_target_map = df_cmp.loc[train_idx].groupby('maintenance_region')['needs_maintenance'].mean()
X_target = df_cmp[numeric_cols].copy()
X_target['vtype_te'] = df_cmp['vehicle_type'].map(type_target_map)
X_target['region_te'] = df_cmp['maintenance_region'].map(region_target_map)
auc_target, acc_target = train_and_evaluate(X_target, "Target Encoding")

# --- 方法C：Frequency Encoding（僅用訓練集計算頻率對照表） ---
type_freq_map = df_cmp.loc[train_idx]['vehicle_type'].value_counts(normalize=True)
region_freq_map = df_cmp.loc[train_idx]['maintenance_region'].value_counts(normalize=True)
X_freq = df_cmp[numeric_cols].copy()
X_freq['vtype_freq'] = df_cmp['vehicle_type'].map(type_freq_map)
X_freq['region_freq'] = df_cmp['maintenance_region'].map(region_freq_map)
auc_freq, acc_freq = train_and_evaluate(X_freq, "Frequency Encoding")

# --- 彙整比較表 ---
result_table = pd.DataFrame({
    '編碼方法': ['One-Hot Encoding', 'Target Encoding', 'Frequency Encoding'],
    'AUC': [auc_onehot, auc_target, auc_freq],
    '準確率': [acc_onehot, acc_target, acc_freq],
    '新增特徵欄位數': [onehot_type.shape[1]+onehot_region.shape[1], 2, 2],
})
print("\n=== 三種編碼方法績效總表 ===")
print(result_table.to_string(index=False))
```

**論文延伸建議**：可進一步在多種不同分類模型（邏輯斯迴歸、隨機森林、梯度提升樹）上重複此比較實驗，觀察「編碼方法的最適選擇是否因模型種類而異」（例如樹狀模型對類別數值的敏感度通常低於線性模型），這種「方法 × 模型」的交叉比較設計，是提升碩士論文貢獻度與嚴謹度的常見手法。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1：資料前處理的順序重要嗎？例如一定要先處理缺失值、還是先處理離群值？**
> A：一般建議順序為：先處理明顯的資料輸入錯誤（如單位錯誤、不可能的數值）→ 處理缺失值 → 處理離群值 → 標準化 → 類別編碼。這個順序的邏輯是：某些離群值判斷（如 IQR 法的四分位數計算）若在有缺失值的狀態下進行，可能因為 pandas 自動排除缺失值而影響四分位數估計的樣本數；標準化必須在處理完離群值後進行，否則統計量（平均數、標準差）會被離群值扭曲。

> [!NOTE]
> **Q2：`fit_transform` 與 `transform` 有什麼差別？為什麼測試集要用 `transform` 而不是 `fit_transform`？**
> A：`fit` 是「從資料中學習統計量」（如平均數、標準差、目標編碼對照表）；`transform` 是「套用已經學到的統計量進行轉換」。訓練集用 `fit_transform`（邊學邊套用）；測試集必須只用 `transform`（套用訓練集學到的統計量，不能讓測試集自己的資料影響統計量的計算），這正是避免資料洩漏的核心操作原則。

> [!NOTE]
> **Q3：One-Hot Encoding 產生的虛擬變數陷阱，實務上一定要處理嗎？**
> A：是否需要移除一欄，取決於後續使用的模型：若使用線性迴歸等對完全共線性敏感的模型，建議移除一欄（`pd.get_dummies(..., drop_first=True)`）；若使用樹狀模型（決策樹、隨機森林、梯度提升樹），這類模型對共線性不敏感，通常不需要特別處理。

> [!NOTE]
> **Q4：VIF 超過門檻後，一定要把變數刪除嗎？**
> A：不一定，刪除變數是最直接的做法，但也可能損失有用資訊。替代方案包括：將高度相關的變數合併為一個複合指標（如主成分分析降維）、改用對共線性較不敏感的模型（如脊迴歸 Ridge Regression，透過正則化緩解係數不穩定的問題）、或者若研究目的只在於預測（非解釋individual係數），共線性對整體預測力影響有限，可以不特別處理。

> [!NOTE]
> **Q5：疊代式插補（IterativeImputer）為什麼在 scikit-learn 中需要額外一行 `enable_iterative_imputer`？**
> A：這是因為 `IterativeImputer` 在 scikit-learn 中仍被歸類為「實驗性功能（Experimental）」，API 未來可能調整，因此官方要求使用者明確「選擇加入（opt-in）」才能使用，這是套件開發者用來提醒使用者「此功能可能不如正式功能穩定」的一種設計慣例，實務使用上不影響功能正確性。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| 缺失值填補（簡單） | `SimpleImputer` | `sklearn.impute` |
| 缺失值填補（KNN） | `KNNImputer` | `sklearn.impute` |
| 缺失值填補（疊代式） | `IterativeImputer` | `sklearn.impute`（需先 `enable_iterative_imputer`） |
| Min-Max 標準化 | `MinMaxScaler` | `sklearn.preprocessing` |
| Z-Score 標準化 | `StandardScaler` | `sklearn.preprocessing` |
| 穩健標準化 | `RobustScaler` | `sklearn.preprocessing` |
| One-Hot 編碼 | `pd.get_dummies` 或 `OneHotEncoder` | `pandas` / `sklearn.preprocessing` |
| VIF 計算 | `variance_inflation_factor` | `statsmodels.stats.outliers_influence` |
| 多變量離群值偵測 | `IsolationForest` | `sklearn.ensemble` |
| 整合式前處理管線 | `Pipeline`、`ColumnTransformer` | `sklearn.pipeline`、`sklearn.compose` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「刪除離群值」 vs 「截尾（Capping）」 | 刪除會減少樣本數；截尾將超出範圍的值直接設為邊界值，保留樣本數但改變其數值，兩者對後續統計檢定的影響不同，需視情境選擇 |
| 「Min-Max 標準化」 vs 「Z-Score 標準化」 | 前者映射到固定 [0,1] 區間、對離群值敏感；後者映射到平均0、標準差1，不受固定區間限制，對離群值相對（但非完全）不敏感 |
| 「One-Hot Encoding」 vs 「Label Encoding」 | One-Hot 不假設類別間有順序關係；Label Encoding 直接把類別轉為 0,1,2,3…數字，會人為賦予類別順序關係，除非類別本身確實有序（如「低/中/高」），否則不建議對名目類別使用 Label Encoding |
| 「訓練集 fit」 vs 「測試集 transform」 | 這是避免資料洩漏最核心的操作原則，任何「需要從資料中學習的統計量」（平均數、標準差、目標編碼對照表）都只能從訓練集學習 |
| 「單變量離群值」 vs 「多變量離群值」 | 單變量只看單一欄位是否超出正常範圍；多變量看多個欄位的「組合」是否合理，兩者偵測到的異常樣本集合通常不完全重疊 |

---

## 附錄D：本週與後續課程週次的關聯

| 後續週次 | 關聯方式 |
|---|---|
| 第 10 週：迴歸與分類基礎 | 本週 VIF 共線性檢查是迴歸分析建模前的必要診斷步驟；本週標準化與編碼方法是邏輯斯迴歸等模型的必要前處理 |
| 第 11 週：非監督學習 | 本週標準化概念對 K-Means 分群尤其重要（若不標準化，數值範圍較大的變數會不成比例主導群聚結果） |
| 第 12 週：決策樹與隨機森林 | 本週示範中已使用隨機森林作為下游評估模型，第 12 週將深入介紹其分割準則與集成學習原理 |
| 第 15、16 週：論文整併實戰 | 本週建立的資料前處理 Pipeline，將直接作為第 15、16 週整合型專論的資料準備模組 |

## 附錄E：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 軍用車輛感測資料（5,000筆） | Hour2主範例／論文方向一、二 | 含缺失值(engine_temp_c 6%、fuel_efficiency 4%)與離群值(vibration、mileage) |
| 彈藥庫溫濕度資料（2,000筆） | 延伸練習一 | 濕度5%缺失，溫度10筆離群值 |
| 無人機零組件庫存資料（1,500筆） | 延伸練習二 | 右偏分配單價，4家供應商類別變數 |

> [!TIP]
> 建議將本週所有程式碼整理成一份 Colab 筆記本並存檔（File → Save a copy in Drive），標註清楚的區塊標題，作為個人的「前處理程式碼工具箱」，未來遇到類似任務時可以直接複製修改使用，不需要每次從頭撰寫。

---

## 參考資料

- [四分位數](https://zh.wikipedia.org/zh-tw/%E5%9B%9B%E5%88%86%E4%BD%8D%E6%95%B8)（維基百科）
- [四分位距](https://zh.wikipedia.org/zh-tw/%E5%9B%9B%E5%88%86%E4%BD%8D%E8%B7%9D)（維基百科）
- [變異數膨脹因子](https://zh.wikipedia.org/zh-tw/%E6%96%B9%E5%B7%AE%E6%93%B4%E5%A4%A7%E5%9B%A0%E5%AD%90)（維基百科）
- [多重共線性](https://zh.wikipedia.org/zh-tw/%E5%A4%9A%E9%87%8D%E5%85%B1%E7%B7%9A%E6%80%A7)（維基百科）
- [scikit-learn 官方文件：資料前處理](https://scikit-learn.org/stable/modules/preprocessing.html)（英文，套件官方文件，示範程式碼之函數詳細參數說明）

---

*下週課程：第 10 週｜統計學習－多變量分析、線性與邏輯斯迴歸（Regression & Logistics）。*
