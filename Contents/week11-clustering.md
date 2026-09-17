# 第 11 週｜非監督學習－分群分析與降維技術（Clustering & Dimensionality Reduction）

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 11 週／共 16 週　**時數**：3 小時
> **本週關鍵字**：K-Means 分群、手肘法、輪廓係數、階層式分群、DBSCAN、主成分分析 PCA、解釋變異比例
> **使用工具**：Google Colab（Python：pandas、NumPy、scikit-learn、scipy）

---

## 0. 本週課程時間分配

| 節次 | 時間 | 內容 |
|---|---|---|
| Hour 1 | 60 分鐘 | 理論深度：K-Means 分群、選 K 值方法、階層式分群、主成分分析 |
| Hour 2 | 60 分鐘 | Python 示範演練：裝備妥善狀態分群、手肘法與輪廓係數、PCA 降維視覺化 |
| Hour 3（前 40 分鐘） | 40 分鐘 | 延伸練習（兩組模擬資料，附完整程式碼）＋ 綜合案例研究 |
| Hour 3（後 20 分鐘） | 20 分鐘 | **碩士論文延伸應用**：兩個可行論文方向之主題定義、方法說明、模擬資料與 Python 實作展示 |

> [!NOTE]
> 建議於 [Google Colab](https://colab.research.google.com/) 開啟新筆記本，依序將本講義程式碼區塊複製貼上執行。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 說明非監督學習與監督學習（第 10 週）的本質差異，理解「沒有標籤答案」情境下的分析目標。
2. 手算 K-Means 分群演算法的疊代邏輯，理解其目標函數（組內平方和）。
3. 運用手肘法與輪廓係數，判斷分群結果中適合的群數 $K$ 。
4. 運用階層式分群建立樹狀圖，理解不同連結法（Linkage）的差異。
5. 運用主成分分析（PCA）進行資料降維，並解讀解釋變異比例與主成分負荷量。
6. 在 Python 中建立完整的分群與降維分析流程，並將分群結果對應回業務意涵。
7. 說出至少兩個可將本週方法延伸為碩士論文題目的具體方向，並理解其對應的進階方法與實作方式。

---

## Hour 1｜理論深度（60 分鐘）

### 1.0 非監督學習導論

第 10 週的迴歸與分類方法，都需要一組「已知答案」的目標變數（維修成本、是否需要維護）才能訓練模型，這類方法統稱**監督學習（Supervised Learning）**。但實務上經常遇到「沒有標準答案」的情境——例如：**我們手上只有數百輛車輛的感測數據，卻不知道該把它們分成幾類妥善狀態、每一類的邊界在哪裡**。**非監督學習（Unsupervised Learning）** 正是處理這類問題的方法家族，核心目標是從資料本身的結構中，挖掘出有意義的樣態，而非預測一個已知的目標。

本週聚焦於非監督學習兩大核心任務：

| 任務 | 目的 | 本週對應小節 |
|---|---|---|
| 分群（Clustering） | 將相似的觀測值歸為同一群，不相似的分開 | 1.1–1.4 |
| 降維（Dimensionality Reduction） | 用較少的變數，保留原始資料中大部分的資訊 | 1.5–1.6 |

---

### 1.1 K-Means 分群演算法

**適用情境**：已知（或假設）資料大致可分為 $K$ 群，需要演算法自動找出每群的中心與各觀測值的歸屬。

**目標函數**：最小化「組內平方和（Within-Cluster Sum of Squares, WCSS）」：

$$
WCSS = \sum_{k=1}^{K}\sum_{i \in C_k} \|x_i-\mu_k\|^2
$$

其中 $\mu_k$ 為第 $k$ 群的中心（群內所有點的平均值）， $C_k$ 為屬於第 $k$ 群的觀測值集合。

**演算法疊代步驟**：

1. 隨機選定 $K$ 個初始群中心。
2. **分配步驟**：將每個觀測值分配給「距離最近」的群中心。
3. **更新步驟**：重新計算每一群「目前所有成員」的平均值，作為新的群中心。
4. 重複步驟 2–3，直到群的歸屬不再改變（收斂）。

> [!IMPORTANT]
> K-Means 的疊代邏輯（分配 → 更新 → 重複），與第 3 週多設施選址問題中 Cooper 位置-分配演算法（分配 → 用 Weiszfeld 法重新選址 → 重複）在數學結構上幾乎完全相同——事實上，若將 K-Means 的「群中心」換成「設施位置」、「組內平方和」換成「加權距離總和」，兩者就是同一類最佳化問題的不同應用情境。這是本課程「作業管理」與「AI 大數據分析」兩階段方法論相互呼應的具體例證，本週 Hour 3 論文方向二會進一步深入探討兩者的連結。

> [!CAUTION]
> K-Means 使用**歐幾里得距離**衡量相似程度，若各變數的量綱差異極大（如里程數以萬為單位、震動值以個位數為單位），數值範圍較大的變數會不成比例地主導分群結果——這正是第 9 週強調「分群前必須標準化」的原因，本週 Hour 2 示範會具體呈現標準化前後分群結果的差異。

延伸閱讀：[K-平均演算法](https://zh.wikipedia.org/zh-tw/K-%E5%B9%B3%E5%9D%87%E7%AE%97%E6%B3%95)（維基百科）。

---

### 1.2 如何選擇群數 K：手肘法與輪廓係數

K-Means 演算法本身不會告訴我們「應該分成幾群」， $K$ 值須由分析者事先指定，以下兩種方法可協助決定合理的 $K$ 值：

**手肘法（Elbow Method）**：讓 $K$ 從 1 逐漸增加，計算每個 $K$ 值下的 WCSS，繪製「 $K$ vs. WCSS」曲線。WCSS 必然隨 $K$ 增加而遞減（群越多、每群越小、組內平方和越小），但遞減速度會在某個 $K$ 值後明顯趨緩，曲線呈現「手肘」形狀的轉折點，該點即為建議的 $K$ 值。

**輪廓係數（Silhouette Coefficient）**：針對每個觀測值 $i$ ，計算：

$$
s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}
$$

其中 $a(i)$ 為 $i$ 與**同群**其他點的平均距離（群內凝聚度，越小越好）， $b(i)$ 為 $i$ 與**最近的其他群**所有點的平均距離（群間分離度，越大越好）。 $s(i)$ 範圍 $[-1,1]$ ，越接近 1 代表該點被分群得越合理。全體觀測值 $s(i)$ 的平均值即為該次分群的整體輪廓係數，可用於比較不同 $K$ 值的分群品質。

> [!TIP]
> 手肘法的「轉折點」判斷帶有主觀性，經常出現曲線並無明顯轉折的情形；輪廓係數則提供一個客觀的數值可直接比較不同 $K$ 。實務上建議**兩種方法並用**：手肘法快速篩選候選範圍，輪廓係數在候選範圍內精確比較——但兩者有時會給出不同的建議 $K$ 值（例如輪廓係數指向 $K=2$ 、但手肘法與業務知識指向 $K=3$），此時**業務脈絡的判斷，往往比純統計指標更重要**，本週 Hour 2 示範會具體遇到這個情境。

延伸閱讀：[輪廓 (聚類)](https://zh.wikipedia.org/zh-tw/%E8%BC%AA%E5%BB%93_(%E8%81%9A%E9%A1%9E))（維基百科）。

---

### 1.3 階層式分群

**適用情境**：不需要事先指定群數，或想要探索資料在不同「精細程度」下的分群結構（如先分成 2 大類，再細分為 5 小類）。

**聚合式階層分群（Agglomerative Hierarchical Clustering）演算法邏輯**：

1. 一開始，每個觀測值各自成一群（$n$ 個觀測值就有 $n$ 群）。
2. 找出「最相似」的兩群，合併為一群。
3. 重複步驟 2，直到所有觀測值合併成一群。
4. 過程中每一次合併的順序與距離，可繪製成**樹狀圖（Dendrogram）**——在樹狀圖上選定一個「切割高度」，就能得到對應該高度的分群結果。

**連結法（Linkage）的選擇**：決定「兩群之間的距離」如何定義：

| 連結法 | 定義 | 特性 |
|---|---|---|
| 單一連結（Single） | 兩群中最近的兩點距離 | 容易產生狹長型的群（鏈狀效應） |
| 完整連結（Complete） | 兩群中最遠的兩點距離 | 傾向產生緊密、大小相近的群 |
| 平均連結（Average） | 兩群所有點對距離的平均 | 介於前兩者之間 |
| 華德法（Ward） | 合併後群內平方和的增量 | 與 K-Means 目標函數精神一致，實務最常用 |

延伸閱讀：[聚類分析](https://zh.wikipedia.org/zh-tw/%E8%81%9A%E9%A1%9E%E5%88%86%E6%9E%90)（維基百科；條目中一併介紹了階層式分群與 DBSCAN 密度分群的原理）。

---

### 1.4 密度分群簡介：DBSCAN

K-Means 與階層式分群（華德法）都隱含假設群集大致呈**凸型（球狀）**，且**強制每個觀測值都必須屬於某一群**——這在「資料中確實存在少數真正的異常值、不應被硬塞進任何一群」的情境下並不理想。**DBSCAN（Density-Based Spatial Clustering of Applications with Noise）** 改以「密度」定義群集：

- 若某點周圍半徑 $\varepsilon$ 內，至少有 `min_samples` 個其他點，該點視為「核心點」，屬於密集區域。
- 由核心點及其密度可達的點共同構成一個群集，群集形狀不受限於凸型。
- **不屬於任何密集區域的點，直接標記為「噪音點（Noise）」**，也就是資料中的離群值——這是 DBSCAN 相對於 K-Means 的關鍵優勢：**異常偵測是密度分群的天然副產物，不需要額外的演算法**。

> [!NOTE]
> DBSCAN 的表現高度仰賴 $\varepsilon$ 與 `min_samples` 兩個參數的設定，且對高維度資料（變數很多時）容易因「維度詛咒」導致密度估計失真，這是實務應用時必須留意的限制，本週 Hour 3 論文方向一會具體示範參數調整對結果的影響。

---

### 1.5 主成分分析（Principal Component Analysis, PCA）

**適用情境**：當自變數數量很多、且變數之間存在相關性時，希望用較少的「綜合指標」保留原始資料中大部分的資訊，同時降低後續分析（如分群、迴歸）的維度與運算複雜度。

**核心概念**：PCA 尋找一組新的座標軸（主成分），每個主成分都是原始變數的線性組合，滿足：

1. **第一主成分（PC1）** 是「資料變異最大」的方向——即所有觀測值投影到這個方向上後，彼此之間的差異最大。
2. **第二主成分（PC2）** 是與 PC1 垂直（正交）、且在此限制下變異最大的方向。
3. 依此類推，直到主成分數量等於原始變數個數。

**數學求解**：主成分即為資料**共變異數矩陣**的**特徵向量**，對應的**特徵值**代表該主成分能解釋的變異量。

$$
\text{解釋變異比例}_k = \frac{\lambda_k}{\sum_j \lambda_j}
$$

其中 $\lambda_k$ 為第 $k$ 大的特徵值。將各主成分的解釋變異比例依序累加，即可判斷「保留前幾個主成分，能保留原始資料多少比例的資訊」。

> [!IMPORTANT]
> PCA 對變數的**尺度極度敏感**——若不先標準化，變異數本來就大的變數（如里程數，數值範圍達數萬）會自動主導第一主成分，即使該變數在業務上不一定最重要。**PCA 前必須先標準化資料**，這是與 K-Means 分群同樣重要、卻經常被忽略的前處理步驟。

延伸閱讀：[主成分分析](https://zh.wikipedia.org/zh-tw/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)（維基百科）。

---

### 1.6 PCA 與分群的結合

PCA 與分群經常搭配使用，原因有二：

1. **視覺化**：高維度資料無法直接繪製散布圖觀察分群效果，但將資料投影到前 2–3 個主成分後，就能在平面或立體空間中視覺化分群結果，直觀檢查分群是否合理。
2. **降低雜訊與計算成本**：後段（解釋變異比例很低）的主成分，經常主要反映隨機雜訊而非真實訊號，捨棄這些成分再進行分群，有時反而能得到更穩定的分群結果，同時大幅降低計算量（尤其在變數數量高達數十、數百個的情境）。

---

### 1.7 方法選擇指南

| 情境 | 建議方法 |
|---|---|
| 已知（或假設）資料大致分為若干個凸型、大小相近的群 | K-Means |
| 想要探索不同精細程度的分群結構、或群數未知 | 階層式分群（樹狀圖） |
| 資料中存在少數真正的異常值，不應被強制分群 | DBSCAN |
| 變數數量多、存在相關性，需要降維或視覺化 | PCA |
| 分群前的變數量綱差異大 | 務必先標準化（第 9 週） |

### 1.8 常用中英文詞彙對照表

| 中文 | 英文 | 中文 | 英文 |
|---|---|---|---|
| 非監督學習 | Unsupervised Learning | 分群 | Clustering |
| 組內平方和 | Within-Cluster Sum of Squares, WCSS | 手肘法 | Elbow Method |
| 輪廓係數 | Silhouette Coefficient | 階層式分群 | Hierarchical Clustering |
| 樹狀圖 | Dendrogram | 連結法 | Linkage |
| 密度分群 | Density-Based Clustering | 主成分分析 | Principal Component Analysis, PCA |
| 解釋變異比例 | Explained Variance Ratio | 主成分負荷量 | Component Loadings |

### 1.9 本週公式總表

| 項目 | 公式 |
|---|---|
| K-Means 目標函數 | $WCSS=\sum_k\sum_{i\in C_k}\lVert x_i-\mu_k\rVert^2$ |
| 輪廓係數 | $s(i)=\dfrac{b(i)-a(i)}{\max(a(i),b(i))}$ |
| PCA 解釋變異比例 | $\lambda_k/\sum_j\lambda_j$ |

### 1.10 各方法的假設條件與限制一覽

| 方法 | 隱含假設 | 主要限制 |
|---|---|---|
| K-Means | 群集大致呈凸型、大小相近；須事先指定 $K$ | 對離群值敏感；每個點都被強制分入某一群，無法識別噪音點 |
| 階層式分群 | 無強制分配假設 | 計算複雜度隨樣本數平方成長，大型資料集計算成本高 |
| DBSCAN | 群集為密度連通區域 | 參數（$\varepsilon$ 、`min_samples`）選擇困難；高維度資料密度估計易失真 |
| PCA | 變數間存在線性相關；資料已標準化 | 只能捕捉線性關係，無法處理非線性的變數關聯結構 |

---

## Hour 2｜Python 示範演練（60 分鐘）

> [!NOTE]
> 本節模擬車輛裝備的感測資料，情境為：後勤處掌握大量車輛的里程、震動、溫度數據，但**不知道這些車輛實際上分屬幾種妥善狀態等級**，希望透過分群分析自動找出合理的分類。

### 2.0 環境設置與模擬資料生成

```python
# ============================================================
# 第11週 Hour 2 示範：裝備妥善狀態分群分析
# ============================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

!apt-get -qq install fonts-noto-cjk > /dev/null 2>&1
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JT', 'Noto Sans CJK TC']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)

# 模擬三種潛在的裝備妥善狀態群集（分析者事先並不知道這個分組，僅能觀測感測數據）
group_sizes = [250, 200, 150]
g1 = np.column_stack([np.random.normal(20000,5000,group_sizes[0]),
                       np.random.normal(1.5,0.4,group_sizes[0]),
                       np.random.normal(85,3,group_sizes[0])])
g2 = np.column_stack([np.random.normal(50000,8000,group_sizes[1]),
                       np.random.normal(3.0,0.5,group_sizes[1]),
                       np.random.normal(89,4,group_sizes[1])])
g3 = np.column_stack([np.random.normal(85000,10000,group_sizes[2]),
                       np.random.normal(5.5,0.7,group_sizes[2]),
                       np.random.normal(94,5,group_sizes[2])])

X_all = np.vstack([g1, g2, g3])
true_labels = np.array([0]*group_sizes[0] + [1]*group_sizes[1] + [2]*group_sizes[2])  # 僅供教學驗證用

df = pd.DataFrame(X_all, columns=['mileage_km', 'vibration_mm_s', 'engine_temp_c'])
print("資料集形狀:", df.shape)
print(df.describe())
```

---

### 2.1 標準化的重要性

```python
# ============================================================
# 2.1 標準化前後分群結果比較（對照1.1節提醒、呼應第9週）
# 說明：此處另外設計一組「里程數範圍重疊、主要靠震動值與溫度才能正確區分」
#       的資料，才能誠實地凸顯標準化的必要性（若三群的里程數本身就分得很開，
#       即使不標準化，分群結果也可能恰好正確，無法凸顯問題）
# ============================================================
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score

np.random.seed(43)
sizes_demo = [250, 200, 150]
d1 = np.column_stack([np.random.normal(45000,8000,sizes_demo[0]),
                        np.random.normal(1.5,0.4,sizes_demo[0]),
                        np.random.normal(85,3,sizes_demo[0])])
d2 = np.column_stack([np.random.normal(48000,8000,sizes_demo[1]),
                        np.random.normal(3.5,0.5,sizes_demo[1]),
                        np.random.normal(89,4,sizes_demo[1])])
d3 = np.column_stack([np.random.normal(50000,8000,sizes_demo[2]),
                        np.random.normal(5.5,0.7,sizes_demo[2]),
                        np.random.normal(94,5,sizes_demo[2])])
X_demo = np.vstack([d1, d2, d3])
true_labels_demo = np.array([0]*sizes_demo[0] + [1]*sizes_demo[1] + [2]*sizes_demo[2])
df_demo = pd.DataFrame(X_demo, columns=['mileage_km', 'vibration_mm_s', 'engine_temp_c'])

# --- 未標準化直接分群 ---
km_raw = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_raw = km_raw.fit_predict(df_demo)
ari_raw = adjusted_rand_score(true_labels_demo, labels_raw)
print(f"未標準化分群結果與真實分組的ARI = {ari_raw:.4f}")

# --- 標準化後分群 ---
scaler_demo = StandardScaler()
X_demo_scaled = scaler_demo.fit_transform(df_demo)
km_scaled = KMeans(n_clusters=3, random_state=42, n_init=10)
labels_scaled = km_scaled.fit_predict(X_demo_scaled)
ari_scaled = adjusted_rand_score(true_labels_demo, labels_scaled)
print(f"標準化後分群結果與真實分組的ARI = {ari_scaled:.4f}")
print(f"\n（ARI越接近1代表分群結果與真實分組越吻合，此處僅供教學驗證，實務上通常沒有真實標籤可對照）")
```

> [!TIP]
> 這個範例中，三個真實群集的**里程數範圍刻意設計為重疊**（平均值 45000／48000／50000，標準差皆達 8000），真正能區分三群的是震動值與溫度；但未標準化時，數值範圍高達數萬的里程數仍會主導距離計算，導致分群結果幾乎與真實分組無關（ARI 接近 0）；標準化後，三個變數才能公平地共同貢獻於分群，ARI 明顯提升。**這也是一個重要的提醒：標準化是否關鍵，取決於「主導距離計算的變數」是否恰好也是「真正決定分群結構的變數」——不能假設數值範圍大的變數一定不重要，也不能假設它一定重要，必須實際比較標準化前後的結果。**

```python
# --- 後續2.2至2.6節的分析，統一使用本節開頭(2.0節)之主資料集df的標準化結果 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
print("主資料集標準化完成，形狀:", X_scaled.shape)
```

---

### 2.2 K-Means 實作與手肘法

```python
# ============================================================
# 2.2 手肘法決定群數K（對照1.2節）
# ============================================================

wcss = []
for k in range(1, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)  # inertia_即為WCSS

plt.figure(figsize=(7, 4.5))
plt.plot(range(1, 9), wcss, marker='o', color='steelblue')
plt.xlabel('群數 K'); plt.ylabel('組內平方和 WCSS')
plt.title('手肘法：K vs. WCSS')
plt.grid(alpha=0.3)
plt.show()

print("WCSS by K:", [round(w, 1) for w in wcss])
print("觀察：K從2到3有明顯下降，K=3之後下降幅度趨緩，K=3是合理的手肘轉折點")
```

---

### 2.3 輪廓係數評估

```python
# ============================================================
# 2.3 輪廓係數比較不同K值（對照1.2節）
# ============================================================
from sklearn.metrics import silhouette_score

sil_scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    sil_scores[k] = silhouette_score(X_scaled, labels)

print("輪廓係數 by K:", {k: round(v, 4) for k, v in sil_scores.items()})
print("\n觀察：K=2的輪廓係數最高，但K=3才符合業務上已知的三種妥善狀態分類，")
print("這是統計指標與業務知識產生分歧時，應優先參考業務脈絡的典型案例（對照1.2節提醒）。")

# --- 最終選定K=3進行分群 ---
km_final = KMeans(n_clusters=3, random_state=42, n_init=10)
final_labels = km_final.fit_predict(X_scaled)
ari_final = adjusted_rand_score(true_labels, final_labels)
print(f"\nK=3最終分群結果與真實分組的ARI = {ari_final:.4f}")

# 還原分群中心到原始尺度，賦予業務意涵
centers_original = scaler.inverse_transform(km_final.cluster_centers_)
print("\n各群中心（原始尺度）：")
for i, c in enumerate(centers_original):
    print(f"  群{i}: 里程={c[0]:.0f}km, 震動={c[1]:.2f}mm/s, 溫度={c[2]:.2f}°C, "
          f"樣本數={sum(final_labels==i)}")
```

---

### 2.4 階層式分群與樹狀圖

```python
# ============================================================
# 2.4 階層式分群（華德法）與樹狀圖（對照1.3節）
# ============================================================
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# --- 華德法分群 ---
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
hc_labels = hc.fit_predict(X_scaled)
sil_hc = silhouette_score(X_scaled, hc_labels)
ari_hc = adjusted_rand_score(true_labels, hc_labels)
print(f"階層式分群（華德法, K=3）: 輪廓係數={sil_hc:.4f}, ARI={ari_hc:.4f}")
print(f"（可與2.3節K-Means的ARI比較，兩種方法在此資料集上表現相近或階層式分群略優）")

# --- 繪製樹狀圖（僅取前50筆樣本繪製，避免圖形過於擁擠）---
Z = linkage(X_scaled[:50], method='ward')
plt.figure(figsize=(10, 5))
dendrogram(Z)
plt.title('階層式分群樹狀圖（前50筆樣本，華德法）')
plt.xlabel('樣本編號'); plt.ylabel('合併距離')
plt.show()
```

---

### 2.5 PCA 降維與視覺化

```python
# ============================================================
# 2.5 PCA降維與視覺化（對照1.5、1.6節）
# ============================================================
from sklearn.decomposition import PCA

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print("各主成分解釋變異比例:", pca.explained_variance_ratio_.round(4))
print("累積解釋變異比例:", np.cumsum(pca.explained_variance_ratio_).round(4))

print("\n主成分負荷量（Loadings）:")
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(3)],
                          index=df.columns)
print(loadings)

# --- 以前兩個主成分視覺化分群結果 ---
plt.figure(figsize=(7, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=final_labels, cmap='viridis', alpha=0.5, s=20)
plt.xlabel(f'PC1（解釋變異{pca.explained_variance_ratio_[0]*100:.1f}%）')
plt.ylabel(f'PC2（解釋變異{pca.explained_variance_ratio_[1]*100:.1f}%）')
plt.title('K-Means分群結果於前兩主成分平面之視覺化')
plt.colorbar(scatter, label='分群標籤')
plt.grid(alpha=0.3)
plt.show()
```

**預期輸出**：PC1 應能解釋超過 80% 的變異，因為里程數、震動值、溫度三者在此資料中都反映同一個潛在的「裝備耗損程度」，PC1 的負荷量在三個原始變數上應呈現方向一致（皆為正值）的權重——**PC1 本質上就是一個「綜合耗損指標」**。視覺化圖形應能清楚看到三個分離良好的群集，驗證 K-Means 分群結果與資料的真實結構相符。

---

### 2.6 分群結果的業務解讀

```python
# ============================================================
# 2.6 將分群結果轉化為業務決策建議
# ============================================================

df['cluster'] = final_labels
summary = df.groupby('cluster').agg(
    平均里程=('mileage_km', 'mean'),
    平均震動=('vibration_mm_s', 'mean'),
    平均溫度=('engine_temp_c', 'mean'),
    數量=('mileage_km', 'count'),
).round(2)
summary = summary.sort_values('平均里程')
summary['建議維護等級'] = ['低度關注', '中度關注', '高度優先維護']
print(summary)
```

> [!NOTE]
> 分群分析的最後一步，永遠是**將統計上的分群結果，轉譯為業務語言與行動建議**——本例中三個群集分別對應「妥善狀態良好」「中度使用」「高度耗損」三種等級，後勤處可依此結果排定巡檢與維保的優先順序，這正是非監督學習從「純數據探索」走向「實務決策支援」的關鍵一步。

---

## Hour 3（前 40 分鐘）｜延伸練習與綜合案例研究

### 3.1 延伸練習一：後勤倉儲分群分析（附完整程式碼）

```python
# ============================================================
# 延伸練習一：依庫存週轉率、平均訂購量、缺貨率為倉儲分群
# ============================================================
np.random.seed(55)

g1 = np.column_stack([np.random.normal(8,1.5,120), np.random.normal(500,100,120),
                        np.random.normal(0.02,0.01,120)])
g2 = np.column_stack([np.random.normal(4,1,100), np.random.normal(200,50,100),
                        np.random.normal(0.08,0.02,100)])
g3 = np.column_stack([np.random.normal(1.5,0.5,80), np.random.normal(80,30,80),
                        np.random.normal(0.15,0.04,80)])
X1 = np.vstack([g1, g2, g3])
df1 = pd.DataFrame(X1, columns=['turnover_rate', 'avg_order_qty', 'stockout_rate'])

scaler1 = StandardScaler()
X1_scaled = scaler1.fit_transform(df1)

wcss1 = [KMeans(n_clusters=k, random_state=42, n_init=10).fit(X1_scaled).inertia_
         for k in range(1, 7)]
print("WCSS:", [round(w, 1) for w in wcss1])

km1 = KMeans(n_clusters=3, random_state=42, n_init=10)
labels1 = km1.fit_predict(X1_scaled)
print(f"輪廓係數(K=3): {silhouette_score(X1_scaled, labels1):.4f}")

centers1 = scaler1.inverse_transform(km1.cluster_centers_)
for i, c in enumerate(centers1):
    print(f"  群{i}: 週轉率={c[0]:.2f}, 平均訂購量={c[1]:.1f}, 缺貨率={c[2]:.3f}")
```

**詳解**：手肘法應在 $K=3$ 附近出現明顯轉折，輪廓係數應接近或超過 0.5（代表分群品質良好）；三群應可分別解讀為「高週轉、低缺貨（管理良好）」「中週轉」「低週轉、高缺貨（需優先檢討補貨策略）」。

---

### 3.2 延伸練習二：裝備健康指標降維分析（附完整程式碼）

```python
# ============================================================
# 延伸練習二：5項相關裝備健康指標的PCA降維
# ============================================================
np.random.seed(66)
n2 = 400

# 5個指標皆由同一個潛在的"裝備健康"因子驅動，彼此高度相關
latent_health = np.random.normal(0, 1, n2)
temp = 88 + 3*latent_health + np.random.normal(0, 1, n2)
vib = 3 - 0.8*latent_health + np.random.normal(0, 0.3, n2)
noise_level = 60 - 5*latent_health + np.random.normal(0, 2, n2)
pressure = 100 + 2*latent_health + np.random.normal(0, 1.5, n2)
efficiency = 85 + 4*latent_health + np.random.normal(0, 1.5, n2)

df2 = pd.DataFrame({'temp': temp, 'vib': vib, 'noise': noise_level,
                      'pressure': pressure, 'efficiency': efficiency})
scaler2 = StandardScaler()
X2_scaled = scaler2.fit_transform(df2)

pca2 = PCA()
pca2.fit(X2_scaled)
print("解釋變異比例:", pca2.explained_variance_ratio_.round(4))
print("累積解釋變異比例:", np.cumsum(pca2.explained_variance_ratio_).round(4))
```

**詳解**：由於 5 個指標皆由同一個潛在因子驅動，PC1 應能解釋超過 85% 的變異——這代表原本需要 5 個變數才能描述的裝備狀態，實際上幾乎可以用**單一綜合指標（PC1）** 完整呈現，是資料維度遠高於真實資訊維度的典型案例。

---

### 3.3 綜合案例研究：建立裝備健康綜合指標與分群預警系統

> [!NOTE]
> 以下是一個虛構但貼近實務的案例，目的是把本週技術串接起來，示範完整的分群與降維分析專案思考過程。

**背景**：某聯兵旅裝備保修處掌握大量車輛的多項感測指標，希望建立一套自動化的裝備狀態分級與預警系統。

**步驟一：資料標準化與探索**（對應 2.0–2.1 節）
保修處整合各項感測數據，確認標準化前後分群結果的差異，避免量綱問題扭曲分析結果。

**步驟二：PCA 建立綜合健康指標**（對應 2.5 節）
若感測指標數量龐大（如超過 10 項），先以 PCA 萃取出 1–2 個能解釋大部分變異的綜合指標，簡化後續分析與溝通的複雜度。

**步驟三：K-Means 分群建立妥善狀態分級**（對應 2.2–2.3 節）
以手肘法與輪廓係數決定合理群數，並確保分群結果與既有的維保分級制度（如「良好／關注／優先」三級）在業務上能夠對應。

**步驟四：轉譯為預警系統**（對應 2.6 節）
將分群結果整合進後勤資訊系統，當新裝備的感測數據被分類至「高度優先維護」群集時，自動觸發巡檢排程建議，並持續追蹤裝備隨時間在不同群集間的移動軌跡（如從「良好」逐漸移動至「關注」），作為預測性維護的早期訊號。

---

## Hour 3（後 20 分鐘）｜碩士論文延伸應用

> [!NOTE]
> 以下兩個方向，示範如何把本週的分群與降維方法延伸為具備研究貢獻的碩士論文題目——核心邏輯是：**K-Means 強制每個觀測值都屬於某一群，無法原生識別真正的異常值；而 K-Means 與多設施選址問題在數學結構上高度相似，這個連結本身就是值得深入探討的研究題材**。每個方向皆包含主題定義、方法說明、模擬資料設計與完整可執行的 Python 實作。

### 3.4 論文方向一：DBSCAN 與 K-Means 於裝備異常偵測之比較研究

**主題定義**：如 1.1 節警告，K-Means 會強制將每個觀測值分入最近的群，即使該點其實是異常值（如感測器故障讀數），也會被硬塞進某一群，僅能透過「距離群心的遠近」間接推測異常程度。本研究比較 K-Means（以距群心距離作為異常分數）與 DBSCAN（原生的噪音點標記機制），在含有真實異常點的裝備感測資料中，偵測異常的精確率與召回率表現，並探討 DBSCAN 參數選擇對結果的敏感度。

**方法說明**：

- **K-Means 異常分數**：分群完成後，計算每個點與其所屬群中心的距離，距離最大的前 $m$ 個點視為候選異常點。
- **DBSCAN 原生噪音標記**：`sklearn` 的 `DBSCAN` 會直接將噪音點標記為 `-1`，不需要額外定義異常分數。
- **參數敏感度分析**：DBSCAN 的 $\varepsilon$（鄰域半徑）與 `min_samples`（核心點最小鄰居數）須嘗試多組組合，觀察其如何影響偵測到的異常點數量與精確率、召回率的取捨。

**模擬資料**：三個正常裝備妥善狀態群集（共 480 筆），另外加入 20 筆分散於特徵空間各處、不成群的真實異常點（模擬感測器故障或極端個案）。

```python
# ============================================================
# 論文方向一：DBSCAN vs. K-Means 異常偵測比較
# ============================================================
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# --- 模擬三個正常群集 + 20筆分散的真實異常點 ---
g1 = np.column_stack([np.random.normal(20000,5000,200), np.random.normal(1.5,0.4,200),
                        np.random.normal(85,3,200)])
g2 = np.column_stack([np.random.normal(50000,8000,180), np.random.normal(3.0,0.5,180),
                        np.random.normal(89,4,180)])
g3 = np.column_stack([np.random.normal(85000,10000,100), np.random.normal(5.5,0.7,100),
                        np.random.normal(94,5,100)])
n_anomaly = 20
anomaly = np.column_stack([
    np.random.uniform(5000,120000,n_anomaly),
    np.random.uniform(0.5,9,n_anomaly),
    np.random.uniform(75,105,n_anomaly),
])

X_all = np.vstack([g1, g2, g3, anomaly])
true_is_anomaly = np.array([0]*480 + [1]*n_anomaly)

df = pd.DataFrame(X_all, columns=['mileage_km', 'vibration_mm_s', 'engine_temp_c'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

def eval_detection(detected, true_anomaly):
    """計算偵測結果的TP/FP/FN與精確率/召回率"""
    tp = np.sum(detected & (true_anomaly==1))
    fp = np.sum(detected & (true_anomaly==0))
    fn = np.sum(~detected & (true_anomaly==1))
    precision = tp/(tp+fp) if (tp+fp)>0 else 0
    recall = tp/(tp+fn) if (tp+fn)>0 else 0
    return tp, fp, fn, precision, recall

# --- 方法A：K-Means，以距群心最遠的20點視為異常 ---
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X_scaled)
distances = np.min(km.transform(X_scaled), axis=1)
km_anomaly_idx = np.argsort(distances)[-n_anomaly:]
km_detected = np.zeros(len(X_all), dtype=bool)
km_detected[km_anomaly_idx] = True
tp_km, fp_km, fn_km, prec_km, rec_km = eval_detection(km_detected, true_is_anomaly)
print(f"=== K-Means（以距群心距離判斷異常）===")
print(f"TP={tp_km}, FP={fp_km}, FN={fn_km}, 精確率={prec_km:.4f}, 召回率={rec_km:.4f}")

# --- 方法B：DBSCAN，原生噪音點標記 ---
dbscan = DBSCAN(eps=0.8, min_samples=10)
db_labels = dbscan.fit_predict(X_scaled)
db_detected = (db_labels == -1)
tp_db, fp_db, fn_db, prec_db, rec_db = eval_detection(db_detected, true_is_anomaly)
print(f"\n=== DBSCAN（原生噪音點標記，eps=0.8, min_samples=10）===")
print(f"TP={tp_db}, FP={fp_db}, FN={fn_db}, 精確率={prec_db:.4f}, 召回率={rec_db:.4f}")
print(f"DBSCAN分群結果: {dict(pd.Series(db_labels).value_counts())}")

# --- 參數敏感度分析：嘗試多組(eps, min_samples)組合 ---
print(f"\n=== DBSCAN參數敏感度分析 ===")
for eps in [0.5, 0.6, 0.7, 0.8]:
    for min_samples in [8, 10, 12]:
        db2 = DBSCAN(eps=eps, min_samples=min_samples)
        labels2 = db2.fit_predict(X_scaled)
        detected2 = (labels2 == -1)
        tp,fp,fn,prec,rec = eval_detection(detected2, true_is_anomaly)
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0
        print(f"eps={eps}, min_samples={min_samples}: "
              f"精確率={prec:.3f}, 召回率={rec:.3f}, F1={f1:.3f}, 標記異常數={detected2.sum()}")
```

**預期輸出**：K-Means 方法的精確率與召回率皆約為 0.60；DBSCAN 在調校後的參數（$\varepsilon=0.8$ ，`min_samples=10`）下，精確率可提升至約 0.87（召回率約 0.65），**整體 F1 分數優於 K-Means**。但參數敏感度分析會顯示：**$\varepsilon$ 稍微調小（如 0.5–0.6），精確率會大幅下降至 0.4–0.6 左右**——DBSCAN 的優勢建立在正確的參數調校之上，這正是它相對於 K-Means「更強大但也更難駕馭」的方法論特性。

> [!IMPORTANT]
> 值得注意的是，本示範中讓 DBSCAN 達到最佳異常偵測表現的 $\varepsilon=0.8$ ，同時也讓原本應該分離的三個正常群集**合併成單一大群**——這揭示了一個重要的研究發現：**DBSCAN 的參數選擇存在「異常偵測敏感度」與「正常群集分離品質」之間的取捨，兩個目標可能無法同時達到最佳**，這是論文延伸應用最值得深入探討的方向。

**論文延伸建議**：可進一步嘗試 HDBSCAN（階層式 DBSCAN 的改良版本，能自動處理不同密度的群集，減少人工調參的負擔）；也可以將此方法實際應用於前置週次（如第 4 週 SPC、第 9 週 Isolation Forest）處理過的感測資料，系統性比較「統計製程管制」「樹狀集成異常偵測」「密度分群異常偵測」三種不同典範方法在同一資料集上的表現差異，建立跨方法論的比較研究。

---

### 3.5 論文方向二：K-Means 分群輔助多設施選址問題之效率研究

**主題定義**：如 1.1 節提醒，K-Means 分群與第 3 週的多設施選址問題（Cooper 位置-分配演算法）在數學結構上高度相似。第 3 週原始方法以**完全隨機的多重啟動策略**尋找較佳解（避免陷入局部最佳解），計算成本較高。本研究提出以 K-Means 分群結果作為初始化，取代隨機多重啟動，驗證此策略能否在維持相近解品質的前提下，大幅縮短求解時間，並探討這個「借用分群方法加速選址問題」的策略在何種情境下最有效。

**方法說明**：

- **傳統做法（隨機多重啟動）**：如第 3 週示範，以 20 組不同的隨機初始位置，各自執行 Cooper 演算法收斂，取其中總成本最低者。
- **K-Means 初始化**：先對需求點座標執行加權 K-Means 分群（以需求量作為樣本權重的近似方式），將分群中心作為 Cooper 演算法的初始設施位置，**只需執行一次**收斂疊代，不需要多重啟動。
- **評估重點**：比較兩種策略的「總成本」與「計算時間」，驗證是否存在「加速但犧牲解品質」的取捨。

**模擬資料**：沿用第 3 週的 10 個需求點資料（5 個彈藥庫補給基地＋5 個通信前哨站）。

```python
# ============================================================
# 論文方向二：K-Means分群輔助多設施選址問題
# ============================================================
import numpy as np
import time
from sklearn.cluster import KMeans

np.random.seed(42)

# --- 沿用第3週10個需求點資料 ---
points = [
    ("B1",30,120,800), ("B2",90,110,650), ("B3",130,60,500),
    ("B4",60,40,900), ("B5",150,130,300),
    ("P1",20,50,400), ("P2",70,90,600), ("P3",100,30,300),
    ("P4",40,10,500), ("P5",120,80,200),
]
coords = np.array([(x,y) for _,x,y,_ in points])
weights = np.array([w for _,_,_,w in points])

def weiszfeld_single(coords, weights, x0, y0, iters=100):
    """單設施韋伯問題疊代解法（對照第3週1.3節）"""
    x, y = x0, y0
    for _ in range(iters):
        d = np.sqrt((coords[:,0]-x)**2+(coords[:,1]-y)**2)
        d = np.where(d<1e-9, 1e-9, d)
        x = np.sum(weights*coords[:,0]/d)/np.sum(weights/d)
        y = np.sum(weights*coords[:,1]/d)/np.sum(weights/d)
    return x, y

def cooper_iterate(coords, weights, facilities, iters=30):
    """給定初始設施位置，執行Cooper位置-分配演算法直到收斂"""
    n = len(coords)
    k = len(facilities)
    for _ in range(iters):
        dists = np.array([[np.hypot(coords[i,0]-f[0],coords[i,1]-f[1]) for f in facilities] for i in range(n)])
        assignment = np.argmin(dists, axis=1)
        new_facilities = facilities.copy()
        for j in range(k):
            mask = assignment==j
            if mask.sum()==0: continue
            nx_,ny_ = weiszfeld_single(coords[mask], weights[mask], *facilities[j], iters=50)
            new_facilities[j] = [nx_, ny_]
        if np.allclose(new_facilities, facilities, atol=1e-6):
            facilities = new_facilities
            break
        facilities = new_facilities
    dists = np.array([[np.hypot(coords[i,0]-f[0],coords[i,1]-f[1]) for f in facilities] for i in range(n)])
    assignment = np.argmin(dists, axis=1)
    total_cost = sum(weights[i]*dists[i,assignment[i]] for i in range(n))
    return facilities, assignment, total_cost

# --- 方法A：傳統隨機多重啟動（對照第3週原始方法）---
def random_restart_strategy(coords, weights, k, n_restarts=20):
    best_cost = np.inf
    best_solution = None
    n = len(coords)
    for _ in range(n_restarts):
        init_idx = np.random.choice(n, k, replace=False)
        facilities = coords[init_idx].astype(float)
        facilities, assignment, cost = cooper_iterate(coords, weights, facilities)
        if cost < best_cost:
            best_cost = cost
            best_solution = (facilities, assignment)
    return best_solution, best_cost

# --- 方法B：K-Means初始化（新方法，僅需一次收斂）---
def kmeans_init_strategy(coords, weights, k):
    # 以權重複製近似加權K-Means：高需求量的點在分群時被賦予更高的影響力
    repeat_factor = (weights/weights.min()).astype(int)
    coords_weighted = np.repeat(coords, repeat_factor, axis=0)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(coords_weighted)
    facilities = km.cluster_centers_.copy()
    facilities, assignment, cost = cooper_iterate(coords, weights, facilities)
    return (facilities, assignment), cost

# --- 比較兩種策略在k=2、k=3下的表現 ---
for k in [2, 3]:
    t0 = time.time()
    sol_random, cost_random = random_restart_strategy(coords, weights, k, n_restarts=20)
    t_random = time.time() - t0

    t0 = time.time()
    sol_kmeans, cost_kmeans = kmeans_init_strategy(coords, weights, k)
    t_kmeans = time.time() - t0

    print(f"\n=== k={k} ===")
    print(f"隨機多重啟動（20次）: 總成本={cost_random:.2f}, 耗時={t_random*1000:.2f}ms")
    print(f"K-Means初始化（1次）: 總成本={cost_kmeans:.2f}, 耗時={t_kmeans*1000:.2f}ms")
    print(f"加速倍數 = {t_random/t_kmeans:.1f}倍，成本差異 = {(cost_kmeans/cost_random-1)*100:.2f}%")
```

**預期輸出**：在 $k=2$ 時，K-Means 初始化策略能找到與隨機多重啟動**完全相同**的最佳解（成本差異 0%），但計算時間僅約為隨機策略的 1/7；在 $k=3$ 時，K-Means 初始化策略雖然計算速度提升超過 15 倍，但最終總成本較隨機多重啟動的最佳解高出約 6%（陷入了品質稍差的局部最佳解）。

> [!IMPORTANT]
> 這個結果揭示了一個誠實而有價值的研究發現：**K-Means 初始化並非在所有情境下都能與隨機多重啟動一樣找到全域最佳解，但它提供了一個極具吸引力的速度與品質折衷方案**。對於需要即時或近即時回應的應用場景（如演訓期間臨時決定新增補給點），犧牲 6% 的解品質、換取 15 倍的求解速度，可能是完全合理的決策；但對於一次性的長期戰略設施規劃，仍建議搭配少量額外的隨機重啟動（混合策略），以更有信心地逼近全域最佳解。

**論文延伸建議**：可將此比較框架擴展至更大規模的需求點網路（如 50、100 個節點），系統性驗證「K-Means 初始化的加速優勢是否隨問題規模擴大而更加顯著」；也可以嘗試「K-Means 初始化＋少量隨機重啟（如 3–5 次）」的混合策略，探討是否能在大幅降低計算時間的同時，仍有效逼近隨機多重啟動的解品質，找出速度與品質之間更精細的帕累托前緣（Pareto Frontier）。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1：K-Means 每次執行的結果都一樣嗎？**
> A：不一定。K-Means 的初始群中心是隨機選定的，不同的初始值可能導致演算法收斂到不同的局部最佳解。實務上（包括 `sklearn` 的預設行為）會以多組不同的隨機初始值分別執行，取其中組內平方和最小者作為最終結果（即 `n_init` 參數），這正是本週 Hour 2 示範中設定 `n_init=10` 的原因。

> [!NOTE]
> **Q2：分群分析要如何知道分群結果「對不對」？**
> A：這是非監督學習與監督學習最本質的差異——沒有「標準答案」可以比對，因此無法用準確率之類的指標直接評估。實務上通常結合三種方式綜合判斷：（1）內部指標如輪廓係數，衡量分群本身的緊密度與分離度；（2）業務合理性，分群結果是否能對應到有意義、可解釋的業務類別；（3）穩定性，改變隨機種子或抽樣不同子集資料，分群結果是否大致穩定一致。

> [!NOTE]
> **Q3：PCA 降維後的主成分，還能還原回原始變數的意義嗎？**
> A：可以透過「主成分負荷量」間接解讀——如本週示範中 PC1 在三個原始變數上都有相近、同方向的負荷量，可解讀為「綜合耗損程度」。但要注意主成分是原始變數的**線性組合**，並非直接對應到某個單一的原始變數，這種「賦予主成分業務意義」的解讀工作，經常需要領域知識輔助，且解讀不一定唯一或客觀。

> [!NOTE]
> **Q4：階層式分群的樹狀圖，「切割高度」要選在哪裡？**
> A：沒有絕對標準，常見做法是尋找樹狀圖中「垂直線最長的一段」進行切割（代表在此高度合併的兩群，彼此差異最大，切開最合理），這在概念上與手肘法尋找轉折點的邏輯相通；也可以直接指定期望的群數 $K$ ，讓演算法自動找出對應的切割高度。

> [!NOTE]
> **Q5：分群分析前，應該先做 PCA 降維，還是直接對原始（標準化後）變數分群？**
> A：兩種做法都常見，取決於情境：若變數數量不多（如本週示範的 3 個變數）、且皆具有明確業務意義，通常建議直接對原始標準化變數分群，保留每個變數的可解釋性；若變數數量龐大（如數十個高度相關的感測指標），先以 PCA 降維萃取出少數主成分，再對主成分分群，能有效降低雜訊干擾與計算成本，但分群結果的業務解讀會變得較為間接。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| K-Means分群 | `KMeans` | `sklearn.cluster` |
| 手肘法WCSS取得 | `KMeans().inertia_` | `sklearn.cluster` |
| 輪廓係數 | `silhouette_score` | `sklearn.metrics` |
| 與真實標籤比較（教學驗證用） | `adjusted_rand_score` | `sklearn.metrics` |
| 階層式分群 | `AgglomerativeClustering` | `sklearn.cluster` |
| 樹狀圖繪製 | `linkage`、`dendrogram` | `scipy.cluster.hierarchy` |
| 密度分群 | `DBSCAN` | `sklearn.cluster` |
| 主成分分析 | `PCA` | `sklearn.decomposition` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「監督學習」（第10週） vs 「非監督學習」（本週） | 前者有明確的目標變數可供訓練與驗證；後者沒有標準答案，僅能從資料結構本身挖掘樣態 |
| 「WCSS」 vs 「輪廓係數」 | WCSS 只衡量群內凝聚度、且必然隨K增加而下降，無法單獨用於選擇K；輪廓係數同時衡量群內凝聚度與群間分離度，可直接比較不同K的優劣 |
| 「K-Means」 vs 「階層式分群」 | K-Means 需事先指定K、計算效率高，適合大型資料集；階層式分群不需事先指定K、能呈現多層次結構，但計算成本隨樣本數平方成長 |
| 「K-Means」 vs 「DBSCAN」 | K-Means 強制每點分入某群、假設群集呈凸型；DBSCAN 允許噪音點存在、能處理任意形狀的群集，但參數選擇較困難 |
| 「主成分」 vs 「原始變數」 | 主成分是原始變數的線性組合，數量上通常遠少於原始變數，但每個主成分不直接對應單一原始變數的業務意義，須透過負荷量間接解讀 |

---

## 附錄D：本週與後續課程週次的關聯

| 後續週次 | 關聯方式 |
|---|---|
| 第 3 週：設施選址規劃 | K-Means 分群與多設施選址（Cooper 演算法）數學結構高度相似，本週 3.5 節已具體展開此連結 |
| 第 9 週：資料前處理 | 標準化是分群與 PCA 分析前不可或缺的前處理步驟，本週 2.1 節具體驗證其必要性 |
| 第 10 週：迴歸與分類 | 監督學習（第10週）與非監督學習（本週）構成機器學習兩大典範，可對照比較適用情境 |
| 第 12 週：決策樹與隨機森林 | 本週分群結果可作為第 12 週監督式分類模型的額外特徵（如將「所屬群集」作為新的類別特徵） |
| 第 15、16 週：論文整併實戰 | 本週建立的分群與降維分析流程，可作為後續整合型專論中「裝備狀態分級」模組的技術基礎 |

## 附錄E：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 裝備妥善狀態資料（600筆，3群） | Hour2主範例 | K=3時ARI約0.89（K-Means）/0.99（階層式） |
| 後勤倉儲週轉資料（300筆，3群） | 延伸練習一 | 週轉率/訂購量/缺貨率三準則分群 |
| 裝備健康指標資料（400筆，5變數） | 延伸練習二 | PC1解釋變異約86% |
| 含異常點裝備資料（500筆，含20筆異常） | 論文方向一 | DBSCAN調校後F1優於K-Means |
| 10節點選址資料（延續第3週） | 論文方向二 | K-Means初始化加速7-15倍 |

---

## 參考資料

- [K-平均演算法](https://zh.wikipedia.org/zh-tw/K-%E5%B9%B3%E5%9D%87%E7%AE%97%E6%B3%95)（維基百科）
- [輪廓 (聚類)](https://zh.wikipedia.org/zh-tw/%E8%BC%AA%E5%BB%93_(%E8%81%9A%E9%A1%9E))（維基百科）
- [聚類分析](https://zh.wikipedia.org/zh-tw/%E8%81%9A%E9%A1%9E%E5%88%86%E6%9E%90)（維基百科）
- [主成分分析](https://zh.wikipedia.org/zh-tw/%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90)（維基百科）

---

*下週課程：第 12 週｜決策樹與隨機森林（Decision Tree & Random Forest）。*
