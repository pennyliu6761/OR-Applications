# 第 10 週｜統計學習－多變量分析、線性與邏輯斯迴歸（Regression & Logistics）

> [!NOTE]
> **課程**：AI 大數據分析　**週次**：第 10 週／共 16 週　**時數**：3 小時
> **本週關鍵字**：最小平方法、決定係數、t 檢定與 F 檢定、高斯－馬可夫假設、邏輯斯迴歸、勝算比、混淆矩陣、ROC 曲線、Ridge／Lasso 正則化
> **使用工具**：Google Colab（Python：pandas、NumPy、statsmodels、scikit-learn）

---

## 0. 本週課程時間分配

| 節次 | 時間 | 內容 |
|---|---|---|
| Hour 1 | 60 分鐘 | 理論深度：簡單／多元線性迴歸、迴歸診斷、邏輯斯迴歸、分類模型評估 |
| Hour 2 | 60 分鐘 | Python 示範演練：延續第 9 週模擬車輛資料，完整迴歸與分類建模流程 |
| Hour 3（前 40 分鐘） | 40 分鐘 | 延伸練習（兩組模擬資料，附完整程式碼）＋ 綜合案例研究 |
| Hour 3（後 20 分鐘） | 20 分鐘 | **碩士論文延伸應用**：兩個可行論文方向之主題定義、方法說明、模擬資料與 Python 實作展示 |

> [!NOTE]
> 建議於 [Google Colab](https://colab.research.google.com/) 開啟新筆記本，依序將本講義程式碼區塊複製貼上執行。本週延續第 9 週建立的模擬車輛感測資料集，若尚未熟悉該資料集的生成邏輯，建議先回顧第 9 週 2.0 節。

---

## 1. 學習目標

完成本週課程後，學員應能夠：

1. 手算簡單線性迴歸的最小平方估計式，並解讀決定係數 $R^2$ 。
2. 運用 t 檢定與 F 檢定判斷迴歸係數與整體模型的統計顯著性。
3. 說明高斯－馬可夫假設的內容，並運用殘差分析診斷迴歸模型是否違反假設。
4. 建立多元線性迴歸模型，解讀各自變數係數的意義。
5. 說明邏輯斯迴歸如何將線性組合轉換為機率，並解讀勝算比（Odds Ratio）。
6. 運用混淆矩陣計算準確率、精確率、召回率、F1 分數，並繪製 ROC 曲線計算 AUC。
7. 在 Python 中建立完整的迴歸與分類建模流程，包含診斷、評估與 Pipeline 整合。
8. 說出至少兩個可將本週方法延伸為碩士論文題目的具體方向，並理解其對應的進階方法與實作方式。

---

## Hour 1｜理論深度（60 分鐘）

### 1.0 迴歸分析導論

迴歸分析是統計學習中最基礎、也最廣泛使用的方法——目標是找出一個或多個**自變數（Independent Variable，又稱特徵、預測變數）**與一個**應變數（Dependent Variable，又稱目標變數）**之間的數量關係。依應變數的型態，迴歸分析可分為兩大類：

| 應變數型態 | 方法 | 本週對應小節 |
|---|---|---|
| 連續型數值（如維修成本、油耗） | 線性迴歸 | 1.1–1.5 |
| 二元類別（如是否故障、是否需要維護） | 邏輯斯迴歸 | 1.6–1.9 |

這兩類方法共享同一套核心概念（係數估計、顯著性檢定、模型診斷），但線性迴歸預測的是「數值本身」，邏輯斯迴歸預測的是「屬於某一類別的機率」——理解這個根本差異，是掌握本週內容的關鍵起點。

延伸閱讀：[迴歸分析](https://zh.wikipedia.org/wiki/%E8%BF%B4%E6%AD%B8%E5%88%86%E6%9E%90)（維基百科）。

---

### 1.1 簡單線性迴歸：最小平方法

**模型設定**：

$$
y_i = \beta_0 + \beta_1 x_i + \varepsilon_i
$$

其中 $\beta_0$ 為截距、 $\beta_1$ 為斜率、 $\varepsilon_i$ 為誤差項（代表模型無法解釋的隨機波動）。

**最小平方法（Ordinary Least Squares, OLS）**：找出一組 $(\hat{\beta}_0, \hat{\beta}_1)$ ，使殘差平方和 $\sum_i(y_i-\hat{y}_i)^2$ 最小。求解可得：

$$
\hat{\beta}_1 = \frac{\sum_i (x_i-\bar{x})(y_i-\bar{y})}{\sum_i(x_i-\bar{x})^2} = \frac{S_{xy}}{S_{xx}}, \qquad \hat{\beta}_0 = \bar{y}-\hat{\beta}_1\bar{x}
$$

**完整手算範例**：某單位蒐集 10 輛車的累積里程數（千公里）與年度維修成本（千元）資料：

| 里程數 $x$ | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 |
|---|---|---|---|---|---|---|---|---|---|---|
| 維修成本 $y$ | 8 | 12 | 15 | 20 | 22 | 28 | 30 | 35 | 40 | 44 |

$$
\bar{x}=55.0, \quad \bar{y}=25.4, \quad S_{xy}=3280.0, \quad S_{xx}=8250.0
$$
$$
\hat{\beta}_1 = \frac{3280.0}{8250.0} = 0.3976, \qquad \hat{\beta}_0 = 25.4-0.3976(55.0) = 3.5333
$$

**迴歸方程式： $\hat{y} = 3.5333+0.3976x$**——即里程數每增加 1 千公里，預估維修成本增加約 397.6 元。

延伸閱讀：[簡單線性迴歸](https://zh.wikipedia.org/zh-tw/%E7%B0%A1%E5%96%AE%E7%B7%9A%E6%80%A7%E8%BF%B4%E6%AD%B8)（維基百科）。

---

### 1.2 迴歸模型評估：決定係數與 F 檢定

**變異數分解**：應變數的總變異，可分解為「模型解釋的變異」與「模型無法解釋的變異」：

$$
\underbrace{\sum_i(y_i-\bar{y})^2}_{SST（總平方和）} = \underbrace{\sum_i(\hat{y}_i-\bar{y})^2}_{SSR（迴歸平方和）} + \underbrace{\sum_i(y_i-\hat{y}_i)^2}_{SSE（殘差平方和）}
$$

**決定係數（Coefficient of Determination）**：

$$
R^2 = \frac{SSR}{SST} = 1-\frac{SSE}{SST}
$$

$R^2$ 衡量「應變數的變異中，能被自變數解釋的比例」，範圍 $[0,1]$ ，越接近 1 代表模型解釋力越強。

**F 檢定**：檢定整體迴歸模型是否顯著（即所有自變數的係數是否至少有一個不為零）：

$$
F = \frac{SSR/p}{SSE/(n-p-1)} \sim F_{(p,\ n-p-1)}
$$

其中 $p$ 為自變數個數， $n$ 為樣本數。

**接續範例**：延續 1.1 節資料，計算得 $SSE=6.35$ 、 $SSR=1304.05$ 、 $SST=1310.40$ ：

$$
R^2 = \frac{1304.05}{1310.40} = 0.9952
$$

即里程數能解釋維修成本 99.52% 的變異，模型配適度極佳。 $F$ 檢定統計量約為 1642.5（$p<0.0001$），代表模型整體高度顯著。

延伸閱讀：[決定係數](https://zh.wikipedia.org/zh-tw/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0)（維基百科）。

---

### 1.3 係數顯著性檢定：t 檢定

除了整體模型的 F 檢定，還須個別檢定**每一個係數**是否顯著不為零——若某自變數的係數檢定不顯著，代表沒有足夠證據顯示該變數真的影響應變數。

$$
t = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)}, \qquad SE(\hat{\beta}_1)=\sqrt{\frac{MSE}{S_{xx}}}, \qquad MSE=\frac{SSE}{n-2}
$$

**接續範例**： $MSE=6.35/8=0.7939$ ， $SE(\hat{\beta}_1)=\sqrt{0.7939/8250}=0.0098$ ：

$$
t = \frac{0.3976}{0.0098} = 40.53
$$

自由度 $n-2=8$ 時，此 $t$ 值對應的 $p$ 值遠小於 0.001，**斜率係數高度顯著**，可以有信心地說里程數確實影響維修成本，而非隨機巧合。

> [!TIP]
> $t$ 檢定與 $F$ 檢定在簡單線性迴歸（只有一個自變數）的情境下，兩者檢定的其實是同一件事（可證明 $t^2=F$），因此結論必然一致；但在多元迴歸中， $F$ 檢定回答「整體模型是否顯著」， $t$ 檢定回答「個別係數是否顯著」，兩者可能給出不同結論（如整體模型顯著，但其中某個別變數不顯著）。

---

### 1.4 迴歸診斷：高斯－馬可夫假設與殘差分析

OLS 估計式要成為「最佳線性不偏估計量（BLUE）」，必須滿足**高斯－馬可夫假設**：

| 假設 | 內容 | 違反時的後果 |
|---|---|---|
| 線性 | 應變數與自變數之間確實存在線性關係 | 模型配適不佳，需考慮轉換變數或非線性模型 |
| 誤差期望值為零 | $E(\varepsilon_i)=0$ | 係數估計有系統性偏誤 |
| 同質變異性（Homoscedasticity） | 誤差變異數 $Var(\varepsilon_i)=\sigma^2$ 對所有觀測值皆相同 | 係數估計仍不偏，但標準誤估計錯誤，顯著性檢定失真 |
| 誤差獨立 | 誤差項之間彼此不相關 | 常見於時間序列資料，標準誤同樣會失真 |
| （若需推論）誤差常態分配 | $\varepsilon_i \sim N(0,\sigma^2)$ | 小樣本時的 $t$ 、 $F$ 檢定不再準確（大樣本時中央極限定理可緩解） |

**殘差分析（Residual Analysis）** 是檢查上述假設是否成立的標準做法：繪製「殘差 vs. 預測值」散布圖，若殘差呈現隨機散布（無明顯型態），代表假設大致成立；若殘差出現喇叭狀（變異隨預測值增加而擴大），代表違反同質變異性假設；若殘差呈現曲線型態，代表模型可能遺漏了非線性關係。

延伸閱讀：[高斯-馬可夫定理](https://zh.wikipedia.org/zh-tw/%E9%AB%98%E6%96%AF-%E9%A6%AC%E5%8F%AF%E5%A4%AB%E5%AE%9A%E7%90%86)（維基百科）。

---

### 1.5 多元線性迴歸：矩陣形式

當自變數超過一個時，模型擴展為：

$$
y_i = \beta_0+\beta_1 x_{i1}+\beta_2 x_{i2}+\cdots+\beta_p x_{ip}+\varepsilon_i
$$

以矩陣表示更為簡潔： $\mathbf{y}=\mathbf{X}\boldsymbol{\beta}+\boldsymbol{\varepsilon}$ ，OLS 解為：

$$
\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}
$$

> [!IMPORTANT]
> 多元迴歸中每個係數的解讀，必須加上「**在控制其他自變數不變的情況下**」這個前提——例如「里程數每增加 1 千公里，維修成本增加 $\hat{\beta}_1$ 元，前提是引擎溫度與震動值維持不變」。這與簡單線性迴歸「單純看 $x$ 與 $y$ 的關係」有本質差異，也是第 9 週 VIF 共線性檢查之所以重要的原因——**若自變數之間高度相關，就無法有意義地「控制其他變數不變」，個別係數的估計會變得不穩定**，這正是本週 Hour 3 論文方向一要深入探討的問題。

延伸閱讀：[線性迴歸](https://zh.wikipedia.org/zh-tw/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8)（維基百科）。

---

### 1.6 邏輯斯迴歸：從線性到分類

**動機**：若應變數是二元類別（0/1），直接套用線性迴歸會產生兩個問題：預測值可能超出 $[0,1]$ 範圍（不符合機率定義）、且違反同質變異性假設。邏輯斯迴歸透過 **Sigmoid（S 型）函數**，將線性組合的輸出「壓縮」到 $(0,1)$ 區間：

$$
P(y=1|x) = \frac{1}{1+e^{-(\beta_0+\beta_1 x)}}
$$

等價地，可以寫成**對數勝算（Log-Odds / Logit）**與自變數呈線性關係：

$$
\ln\left(\frac{P(y=1)}{1-P(y=1)}\right) = \beta_0+\beta_1 x
$$

**參數估計**：邏輯斯迴歸不像線性迴歸有封閉解，須透過**最大概似估計法（Maximum Likelihood Estimation, MLE）**疊代求解，找出使觀測資料出現機率最大的參數組合。

**完整手算範例**：某單位蒐集 18 件裝備的震動值與是否需要維護的資料，以邏輯斯迴歸估計：

$$
\hat{\beta}_0=-7.3252, \qquad \hat{\beta}_1=2.1031
$$

即 $P(\text{需要維護}) = \dfrac{1}{1+e^{-(-7.3252+2.1031x)}}$ ，其中 $x$ 為震動值。兩個係數皆顯著（$p<0.05$）。

---

### 1.7 邏輯斯迴歸係數解讀：勝算比

邏輯斯迴歸係數 $\beta_1$ 本身不是直接的機率變化量，而是「自變數每增加 1 單位，對數勝算的變化量」。取指數轉換後得到**勝算比（Odds Ratio, OR）**：

$$
OR = e^{\beta_1}
$$

**接續範例**： $OR=e^{2.1031}=8.19$ ，解讀為：**震動值每增加 1 mm/s，該裝備「需要維護」的勝算（Odds），變為原本的 8.19 倍**。

> [!NOTE]
> 「勝算（Odds）」與「機率（Probability）」是不同的概念：機率是「發生的次數／總次數」；勝算是「發生的次數／不發生的次數」，即 $Odds=P/(1-P)$ 。 $OR>1$ 代表該自變數增加會提高事件發生的勝算， $OR<1$ 代表降低， $OR=1$ 代表無影響。勝算比不能直接解讀為「機率變為幾倍」，這是初學者最常見的誤用。

延伸閱讀：[邏輯迴歸](https://zh.wikipedia.org/zh-tw/%E9%82%8F%E8%BC%AF%E8%BF%B4%E6%AD%B8)、[廣義線性模型](https://zh.wikipedia.org/zh-tw/%E5%BB%A3%E7%BE%A9%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B)（維基百科；邏輯斯迴歸是廣義線性模型的一個特例）。

---

### 1.8 分類模型評估：混淆矩陣與衍生指標

**混淆矩陣（Confusion Matrix）**：

| | 預測為正 | 預測為負 |
|---|---|---|
| 實際為正 | 真陽性 TP | 偽陰性 FN |
| 實際為負 | 偽陽性 FP | 真陰性 TN |

**衍生指標**：

$$
\text{準確率（Accuracy）} = \frac{TP+TN}{TP+TN+FP+FN}
$$
$$
\text{精確率（Precision）} = \frac{TP}{TP+FP}, \qquad \text{召回率（Recall，又稱敏感度）} = \frac{TP}{TP+FN}
$$
$$
F_1 = \frac{2\times Precision\times Recall}{Precision+Recall}
$$

> [!CAUTION]
> **準確率在類別不平衡的資料中極具誤導性**：若 100 件裝備中只有 5 件真的故障，一個「無論如何都預測不故障」的模型，準確率高達 95%，卻毫無實用價值（召回率為 0%，一件故障都抓不到）。**精確率與召回率之間通常存在取捨關係**：提高召回率（抓到更多真陽性）經常伴隨精確率下降（誤判更多假陽性），本週 Hour 3 論文方向二會具體示範這個取捨在稀有故障預測情境下的實務意涵。

延伸閱讀：[混淆矩陣](https://zh.wikipedia.org/zh-tw/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%99%A3)、[靈敏度和特異度](https://zh.wikipedia.org/zh-tw/%E9%9D%88%E6%95%8F%E5%BA%A6%E5%92%8C%E7%89%B9%E7%95%B0%E5%BA%A6)（維基百科）。

---

### 1.9 ROC 曲線與 AUC

混淆矩陣的各項指標，都建立在「分類門檻（Threshold）」固定為 0.5 的前提下——但實務上可以調整這個門檻。**ROC 曲線（Receiver Operating Characteristic Curve）** 呈現門檻從 0 到 1 變動時，「真陽性率（即召回率）」與「偽陽性率」之間的權衡關係：

$$
\text{偽陽性率（FPR）} = \frac{FP}{FP+TN}
$$

**AUC（Area Under the Curve）** 為 ROC 曲線下的面積，範圍 $[0,1]$ ： $AUC=1$ 代表完美分類器， $AUC=0.5$ 代表與隨機猜測無異，AUC 的優點是**不受分類門檻選擇的影響**，適合用於比較不同模型的整體區辨能力。

延伸閱讀：[ROC曲線](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF)（維基百科）。

---

### 1.10 方法選擇指南

| 情境 | 建議方法 |
|---|---|
| 應變數為連續數值 | 線性迴歸 |
| 應變數為二元類別 | 邏輯斯迴歸 |
| 需要判斷模型整體是否顯著 | F 檢定 |
| 需要判斷個別自變數是否顯著 | t 檢定 |
| 自變數之間存在高度相關 | 檢查 VIF（第 9 週），必要時改用 Ridge／Lasso（本週 Hour 3） |
| 分類目標的類別比例懸殊（稀有事件） | 避免僅看準確率，改用精確率、召回率、F1、AUC，並考慮不平衡資料處理技術（本週 Hour 3） |

### 1.11 常用中英文詞彙對照表

| 中文 | 英文 | 中文 | 英文 |
|---|---|---|---|
| 最小平方法 | Ordinary Least Squares, OLS | 決定係數 | Coefficient of Determination, $R^2$ |
| 殘差 | Residual | 高斯－馬可夫假設 | Gauss-Markov Assumptions |
| 同質變異性 | Homoscedasticity | 最大概似估計法 | Maximum Likelihood Estimation, MLE |
| 勝算比 | Odds Ratio | 混淆矩陣 | Confusion Matrix |
| 精確率 | Precision | 召回率 | Recall |
| ROC 曲線 | Receiver Operating Characteristic Curve | 曲線下面積 | Area Under the Curve, AUC |

### 1.12 本週公式總表

| 項目 | 公式 |
|---|---|
| 簡單迴歸斜率 | $\hat{\beta}_1=S_{xy}/S_{xx}$ |
| 決定係數 | $R^2=SSR/SST$ |
| $t$ 統計量 | $\hat{\beta}_1/SE(\hat{\beta}_1)$ |
| 多元迴歸矩陣解 | $\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$ |
| 邏輯斯函數 | $P(y=1\vert x)=1/(1+e^{-(\beta_0+\beta_1x)})$ |
| 勝算比 | $OR=e^{\beta_1}$ |
| 精確率／召回率 | $TP/(TP+FP)$／$TP/(TP+FN)$ |

### 1.13 各方法的假設條件與限制一覽

| 方法 | 隱含假設 | 主要限制 |
|---|---|---|
| 線性迴歸 | 高斯－馬可夫假設（1.4 節） | 對離群值敏感；假設關係為線性，無法捕捉複雜非線性樣態 |
| 邏輯斯迴歸 | 對數勝算與自變數呈線性關係；觀測值獨立 | 類別嚴重不平衡時，預設 0.5 門檻容易失真（1.8 節提醒） |
| $R^2$ | 樣本數足夠、模型設定正確 | 加入更多自變數必然使 $R^2$ 上升（即使該變數無意義），須改用調整後 $R^2$ 校正 |

---

## Hour 2｜Python 示範演練（60 分鐘）

> [!NOTE]
> 本節延續第 9 週建立的模擬車輛感測資料集，新增「維修成本」連續變數，供線性迴歸示範使用；「是否需要維護」二元變數則沿用第 9 週設定，供邏輯斯迴歸示範使用。

### 2.0 環境設置與模擬資料生成

```python
# ============================================================
# 第10週 Hour 2 示範：迴歸與分類建模完整流程
# 情境：延續第9週模擬車輛感測資料，新增維修成本變數
# ============================================================
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib

!apt-get -qq install fonts-noto-cjk > /dev/null 2>&1
matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JT', 'Noto Sans CJK TC']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(42)
n = 1000

vehicle_type = np.random.choice(
    ['戰甲車', '輪型運輸車', '工程車', '救護車'], size=n, p=[0.35, 0.4, 0.15, 0.1]
)
mileage_km = np.random.normal(45000, 15000, n).clip(500, None)
engine_temp_c = np.random.normal(88, 6, n)
vibration_mm_s = np.random.gamma(shape=2.0, scale=1.5, size=n)
fuel_efficiency_km_l = 12 - 0.00008*mileage_km + np.random.normal(0, 0.8, n)

# 新增：維修成本（連續變數），與里程、震動、引擎溫度存在線性關係，供線性迴歸示範
maintenance_cost = (2000 + 0.015*mileage_km + 150*vibration_mm_s
                     + 30*(engine_temp_c-88) + np.random.normal(0, 400, n))
maintenance_cost = np.clip(maintenance_cost, 500, None)

# 沿用第9週邏輯：是否需要維護（二元變數），供邏輯斯迴歸示範
risk_score = (0.00002*mileage_km + 0.03*engine_temp_c
              + 0.15*vibration_mm_s + np.random.normal(0, 0.5, n))
threshold = np.percentile(risk_score, 70)
needs_maintenance = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    'vehicle_type': vehicle_type,
    'mileage_km': mileage_km.round(1),
    'engine_temp_c': engine_temp_c.round(2),
    'vibration_mm_s': vibration_mm_s.round(3),
    'fuel_efficiency_km_l': fuel_efficiency_km_l.round(2),
    'maintenance_cost': maintenance_cost.round(0),
    'needs_maintenance': needs_maintenance,
})

print("資料集形狀:", df.shape)
print(df.describe())
```

---

### 2.1 探索性資料分析與相關性

```python
# ============================================================
# 2.1 探索性資料分析：散布圖與相關係數矩陣
# ============================================================

# 相關係數矩陣，快速掃描哪些變數與維修成本較相關
numeric_cols = ['mileage_km', 'engine_temp_c', 'vibration_mm_s', 'maintenance_cost']
print(df[numeric_cols].corr().round(3))

# 繪製散布圖矩陣的簡化版：里程數 vs 維修成本
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df['mileage_km'], df['maintenance_cost'], alpha=0.3, s=15)
ax.set_xlabel('里程數 (km)')
ax.set_ylabel('維修成本 (元)')
ax.set_title('里程數 vs. 維修成本 散布圖')
plt.grid(alpha=0.3)
plt.show()
```

---

### 2.2 簡單線性迴歸實作

```python
# ============================================================
# 2.2 簡單線性迴歸：以里程數預測維修成本（對照1.1節）
# ============================================================

X_simple = sm.add_constant(df['mileage_km'])
y = df['maintenance_cost']
model_simple = sm.OLS(y, X_simple).fit()
print(model_simple.summary())
```

> [!TIP]
> `statsmodels` 的 `.summary()` 輸出，把本週理論介紹的所有統計量（係數、標準誤、 $t$ 值、 $p$ 值、 $R^2$ 、 $F$ 統計量）整合在同一張表格中，是撰寫論文結果章節時最常直接引用的標準輸出格式。

---

### 2.3 多元線性迴歸與係數解讀

```python
# ============================================================
# 2.3 多元線性迴歸（對照1.5節）
# ============================================================

X_multi = sm.add_constant(df[['mileage_km', 'engine_temp_c', 'vibration_mm_s']])
model_multi = sm.OLS(y, X_multi).fit()
print(model_multi.summary())

print(f"\n調整後R2: {model_multi.rsquared_adj:.4f}")
print(f"（調整後R2會對自變數個數進行懲罰，比較不同自變數個數的模型時應使用此指標而非原始R2）")
```

---

### 2.4 迴歸診斷：殘差分析與共線性檢查

```python
# ============================================================
# 2.4 迴歸診斷（對照1.4節，並延續第9週VIF概念）
# ============================================================
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan

# --- 殘差 vs. 預測值散布圖：檢查同質變異性假設 ---
fitted = model_multi.fittedvalues
residuals = model_multi.resid

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].scatter(fitted, residuals, alpha=0.3, s=15)
axes[0].axhline(0, color='red', linestyle='--')
axes[0].set_xlabel('預測值'); axes[0].set_ylabel('殘差')
axes[0].set_title('殘差 vs. 預測值圖')

# --- Q-Q圖：檢查殘差是否近似常態分配 ---
sm.qqplot(residuals, line='45', ax=axes[1])
axes[1].set_title('殘差常態Q-Q圖')
plt.tight_layout()
plt.show()

# --- Breusch-Pagan檢定：正式檢定是否違反同質變異性假設 ---
bp_test = het_breuschpagan(residuals, X_multi)
print(f"Breusch-Pagan檢定: LM統計量={bp_test[0]:.4f}, p值={bp_test[1]:.4f}")
print("p值 > 0.05 代表沒有足夠證據拒絕「同質變異性」假設")

# --- VIF共線性檢查（呼應第9週1.5節）---
vif_data = pd.DataFrame()
vif_data['特徵'] = X_multi.columns
vif_data['VIF'] = [variance_inflation_factor(X_multi.values, i) for i in range(X_multi.shape[1])]
print("\nVIF檢查:")
print(vif_data)
```

---

### 2.5 邏輯斯迴歸實作

```python
# ============================================================
# 2.5 邏輯斯迴歸：預測是否需要維護（對照1.6、1.7節）
# ============================================================
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_clf = df[['mileage_km', 'engine_temp_c', 'vibration_mm_s']]
y_clf = df['needs_maintenance']

# 切分訓練/測試集（呼應第9週資料洩漏防範原則）
Xtr, Xte, ytr, yte = train_test_split(X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf)

clf = LogisticRegression()
clf.fit(Xtr, ytr)

print("邏輯斯迴歸係數:")
for name, coef in zip(X_clf.columns, clf.coef_[0]):
    print(f"  {name}: 係數={coef:.4f}, 勝算比={np.exp(coef):.4f}")
print(f"截距: {clf.intercept_[0]:.4f}")
```

---

### 2.6 分類模型評估：混淆矩陣與 ROC 曲線

```python
# ============================================================
# 2.6 分類模型評估（對照1.8、1.9節）
# ============================================================
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                               recall_score, f1_score, roc_auc_score, roc_curve)

pred = clf.predict(Xte)
proba = clf.predict_proba(Xte)[:, 1]

print("=== 混淆矩陣 ===")
cm = confusion_matrix(yte, pred)
print(cm)

print(f"\n準確率 = {accuracy_score(yte, pred):.4f}")
print(f"精確率 = {precision_score(yte, pred):.4f}")
print(f"召回率 = {recall_score(yte, pred):.4f}")
print(f"F1分數 = {f1_score(yte, pred):.4f}")
print(f"AUC = {roc_auc_score(yte, proba):.4f}")

# 繪製ROC曲線
fpr, tpr, thresholds = roc_curve(yte, proba)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'ROC曲線 (AUC={roc_auc_score(yte,proba):.3f})', color='steelblue')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='隨機猜測基準線')
plt.xlabel('偽陽性率 (FPR)'); plt.ylabel('真陽性率 (TPR)')
plt.title('ROC 曲線')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

### 2.7 整合為完整 Pipeline

```python
# ============================================================
# 2.7 整合為完整建模Pipeline（結合第9週前處理與本週建模）
# ============================================================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

numeric_features = ['mileage_km', 'engine_temp_c', 'vibration_mm_s']
categorical_features = ['vehicle_type']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numeric_features),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
])

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression()),
])

X_full = df[numeric_features + categorical_features]
Xtr2, Xte2, ytr2, yte2 = train_test_split(X_full, y_clf, test_size=0.25, random_state=42, stratify=y_clf)

full_pipeline.fit(Xtr2, ytr2)
pred2 = full_pipeline.predict(Xte2)
print(f"整合Pipeline準確率 = {accuracy_score(yte2, pred2):.4f}")
print(f"整合Pipeline AUC = {roc_auc_score(yte2, full_pipeline.predict_proba(Xte2)[:,1]):.4f}")
```

> [!TIP]
> 這個 Pipeline 把第 9 週的前處理（標準化、One-Hot 編碼）與本週的邏輯斯迴歸建模整合為單一物件，未來有新資料進來時，只需要呼叫 `full_pipeline.predict(新資料)`，前處理與建模會自動依序套用，不需要重複撰寫每一個步驟——這是機器學習專案從「示範程式碼」邁向「可部署系統」的關鍵一步。

---

## Hour 3（前 40 分鐘）｜延伸練習與綜合案例研究

### 3.1 延伸練習一：彈藥庫月耗損量多元迴歸（附完整程式碼）

```python
# ============================================================
# 延伸練習一：彈藥庫儲存條件與月耗損量之多元迴歸
# ============================================================
np.random.seed(21)
n1 = 200

storage_temp = np.random.normal(24, 4, n1)
humidity = np.random.normal(50, 10, n1)
storage_age_years = np.random.uniform(1, 20, n1)
monthly_loss = (5 + 0.8*storage_temp + 0.3*humidity + 1.2*storage_age_years
                + np.random.normal(0, 8, n1))

df1 = pd.DataFrame({
    'storage_temp': storage_temp, 'humidity': humidity,
    'storage_age_years': storage_age_years, 'monthly_loss': monthly_loss,
})

X1 = sm.add_constant(df1[['storage_temp', 'humidity', 'storage_age_years']])
model1 = sm.OLS(df1['monthly_loss'], X1).fit()
print(model1.summary())
```

**詳解**：三個自變數的係數皆應顯著（$p<0.001$）， $R^2$ 約為 0.48，代表儲存溫度、濕度、儲存年限合計能解釋約 48% 的月耗損量變異；儲存年限的係數最大，顯示裝備老化對耗損量的邊際影響最為顯著。

---

### 3.2 延伸練習二：通信裝備故障邏輯斯迴歸（附完整程式碼）

```python
# ============================================================
# 延伸練習二：通信裝備訊號強度、使用月數與故障之邏輯斯迴歸
# ============================================================
np.random.seed(31)
n2 = 300

signal_strength = np.random.normal(70, 15, n2)
age_months = np.random.uniform(1, 60, n2)
logit = -3 + 0.04*age_months - 0.02*signal_strength
prob = 1/(1+np.exp(-logit))
failure = (np.random.random(n2) < prob).astype(int)

df2 = pd.DataFrame({'signal_strength': signal_strength, 'age_months': age_months, 'failure': failure})
print(f"故障比例: {df2['failure'].mean()*100:.2f}%")

X2 = df2[['signal_strength', 'age_months']]
y2 = df2['failure']
Xtr2, Xte2, ytr2, yte2 = train_test_split(X2, y2, test_size=0.3, random_state=42, stratify=y2)

clf2 = LogisticRegression().fit(Xtr2, ytr2)
pred2 = clf2.predict(Xte2)
print(f"準確率 = {accuracy_score(yte2, pred2):.4f}")
print(f"AUC = {roc_auc_score(yte2, clf2.predict_proba(Xte2)[:,1]):.4f}")
print(f"係數與勝算比:")
for name, coef in zip(X2.columns, clf2.coef_[0]):
    print(f"  {name}: 係數={coef:.4f}, 勝算比={np.exp(coef):.4f}")
```

**詳解**：`age_months` 的勝算比應大於 1（使用月數越長、故障勝算越高），`signal_strength` 的勝算比應小於 1（訊號強度越高、故障勝算越低），兩者方向皆符合直覺，AUC 應落在 0.8 附近，代表模型有良好的區辨能力。

---

### 3.3 綜合案例研究：建立裝備維修成本與故障風險雙軌預測系統

> [!NOTE]
> 以下是一個虛構但貼近實務的案例，目的是把本週技術串接起來，示範完整的迴歸與分類建模專案思考過程。

**背景**：某聯兵旅後勤處欲建立一套雙軌預測系統，同時掌握「預期維修成本」與「故障風險」兩項資訊，作為維保預算編列與優先排修的依據。

**步驟一：資料整合與探索**（對應 2.0–2.1 節）
後勤處整合車輛感測資料與歷史維修記錄，計算各變數與兩項目標（維修成本、是否需要維護）的相關性，初步篩選候選自變數。

**步驟二：建立線性迴歸模型預測維修成本**（對應 2.2–2.4 節）
以多元線性迴歸建立維修成本預測模型，並透過殘差分析與 VIF 檢查確認模型符合迴歸假設、無嚴重共線性問題，作為年度維保預算編列的量化依據。

**步驟三：建立邏輯斯迴歸模型預測故障風險**（對應 2.5–2.6 節）
以邏輯斯迴歸建立故障風險預測模型，透過混淆矩陣與 ROC 曲線評估模型效能，並依指揮官對「寧可誤判、不可漏判」的風險偏好，調整分類門檻（而非機械式套用 0.5）。

**步驟四：整合為決策支援系統**（對應 2.7 節）
將兩套模型分別封裝為 Pipeline，整合進後勤資訊系統，每月自動產出「預期維修成本排行」與「高故障風險裝備清單」雙軌報表，供指揮官排定維保優先順序。

---

## Hour 3（後 20 分鐘）｜碩士論文延伸應用

> [!NOTE]
> 以下兩個方向，示範如何把本週的迴歸與分類方法延伸為具備研究貢獻的碩士論文題目——核心邏輯是：**當自變數之間高度相關時，OLS 的係數估計會變得極不穩定，甚至出現違反直覺的正負號；當分類目標的類別比例嚴重失衡時，預設的 0.5 分類門檻與準確率指標會嚴重誤導決策**。每個方向皆包含主題定義、方法說明、模擬資料設計與完整可執行的 Python 實作。

### 3.4 論文方向一：Ridge／Lasso 正則化迴歸於共線性資料之應用

**主題定義**：軍用裝備的多項監測指標經常高度相關（如累積里程數與累積運轉時數，兩者都反映裝備使用程度），如 1.5 節提醒，此時 OLS 估計的個別係數會變得不穩定，甚至出現正負號違反直覺的情形。本研究比較 OLS、Ridge（L2 正則化）、Lasso（L1 正則化）三種方法，在高度共線性資料上的係數穩定度與預測表現，驗證正則化方法是否能有效緩解共線性造成的估計不穩定問題。

**方法說明**：

- **Ridge 迴歸（嶺迴歸）**：在 OLS 的目標函數中加入係數平方和的懲罰項 $\lambda\sum_j\beta_j^2$ ，傾向讓所有相關係數的估計值「平均分攤」，而不會讓某個係數異常放大、另一個異常縮小（甚至變號）。
- **Lasso 迴歸**：加入係數絕對值和的懲罰項 $\lambda\sum_j|\beta_j|$ ，除了穩定係數估計，還具有**自動變數選擇**的效果——會把冗餘變數的係數直接壓縮至恰好為零。
- **係數穩定度驗證**：以自助抽樣法（Bootstrap）對訓練資料重複抽樣多次，分別重新配適 OLS、Ridge、Lasso，比較同一係數在不同次抽樣下的標準差，標準差越小代表估計越穩定。

**模擬資料**：模擬里程數與運轉時數兩個高度相關（相關係數約 0.99）的變數，其中僅里程數真正影響維修成本，運轉時數為多餘的共線變數。

```python
# ============================================================
# 論文方向一：Ridge/Lasso正則化迴歸處理共線性
# ============================================================
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)
n = 300

# --- 模擬高度共線資料：operating_hours與mileage高度相關(相關係數約0.99) ---
mileage = np.random.normal(45000, 15000, n).clip(500, None)
operating_hours = mileage * 0.025 + np.random.normal(0, 50, n)
vibration = np.random.gamma(2.0, 1.5, n)
# 真實模型：維修成本僅受mileage與vibration影響，operating_hours為多餘變數
maintenance_cost = 2000 + 0.02*mileage + 100*vibration + np.random.normal(0, 400, n)

X = pd.DataFrame({'mileage': mileage, 'operating_hours': operating_hours, 'vibration': vibration})
y = np.asarray(maintenance_cost)

print(f"mileage 與 operating_hours 相關係數 = {X['mileage'].corr(X['operating_hours']):.4f}")

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
X_vif = sm.add_constant(X)
vif = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
print(f"VIF: {dict(zip(X_vif.columns, [round(v,2) for v in vif]))}")

# --- 標準化後比較OLS、Ridge、Lasso係數 ---
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

ols = LinearRegression().fit(Xtr, ytr)
ridge = Ridge(alpha=50.0).fit(Xtr, ytr)
lasso = Lasso(alpha=5.0).fit(Xtr, ytr)

print("\n=== 標準化後係數比較 ===")
print(f"OLS:   {dict(zip(X.columns, ols.coef_.round(2)))}")
print(f"Ridge: {dict(zip(X.columns, ridge.coef_.round(2)))}")
print(f"Lasso: {dict(zip(X.columns, lasso.coef_.round(2)))}")

# --- 係數穩定度測試：50次自助抽樣，比較係數的標準差 ---
print(f"\n=== 係數穩定度測試（50次Bootstrap重抽樣）===")
np.random.seed(1)
ols_coefs, ridge_coefs, lasso_coefs = [], [], []
for i in range(50):
    idx = np.random.choice(len(Xtr), len(Xtr), replace=True)
    Xb, yb = Xtr.iloc[idx], ytr[idx]
    ols_coefs.append(LinearRegression().fit(Xb, yb).coef_)
    ridge_coefs.append(Ridge(alpha=50.0).fit(Xb, yb).coef_)
    lasso_coefs.append(Lasso(alpha=5.0).fit(Xb, yb).coef_)

ols_coefs, ridge_coefs, lasso_coefs = np.array(ols_coefs), np.array(ridge_coefs), np.array(lasso_coefs)
for i, col in enumerate(X.columns):
    print(f"{col}: OLS標準差={ols_coefs[:,i].std():.3f}, "
          f"Ridge標準差={ridge_coefs[:,i].std():.3f}, Lasso標準差={lasso_coefs[:,i].std():.3f}")
```

**預期輸出**：VIF 檢查應顯示 `mileage`、`operating_hours` 的 VIF 皆超過 50（遠超過第 9 週介紹的 >5 嚴重共線性門檻）。**OLS 估計的 `operating_hours` 係數會出現負值**（違反直覺——運轉時數增加照理不應降低預期維修成本），這正是共線性導致係數估計不穩定的典型病徵；Ridge 的兩個相關係數則會呈現較合理、方向一致的正值。Bootstrap 穩定度測試應顯示：**Ridge 係數的標準差遠小於 OLS**（如降低超過 10 倍），Lasso 則會把其中一個共線變數的係數**直接壓縮為零**，等同於自動判斷該變數是多餘的。

> [!IMPORTANT]
> 這個結果對論文寫作有重要的方法論意涵：**若您的迴歸模型出現係數正負號違反理論預期，第一步不該急著重新詮釋結果，而應該先檢查 VIF 是否過高**。共線性不會讓模型的整體預測力變差（OLS、Ridge、Lasso 的預測誤差通常相近），但會讓「個別係數的解釋」變得不可靠，這對於強調係數解釋（而非純粹預測）的政策分析類論文尤其關鍵。

**論文延伸建議**：可進一步以交叉驗證（Cross-Validation）系統性搜尋最適的正則化強度 $\lambda$ ，取代本示範中直接指定的固定值；也可以將此方法應用於第 9 週資料前處理單元中曾出現的多重感測指標資料，探討軍用裝備感測資料中還有哪些指標組合存在類似的共線性問題。

---

### 3.5 論文方向二：不平衡資料分類技術於稀有故障預測之應用

**主題定義**：如 1.8 節提醒，當故障是稀有事件（如僅 8% 的裝備會發生嚴重故障）時，標準邏輯斯迴歸在預設 0.5 門檻下，容易系統性地低估故障機率、大量漏判真正會故障的裝備——這在軍事後勤情境中是不可接受的（寧可誤判、不可漏判關鍵裝備）。本研究比較基準邏輯斯迴歸、類別權重調整（Class Weighting）、SMOTE 過採樣三種方法，在稀有故障預測情境下的召回率與精確率表現，驗證何種方法最適合「不能漏判」的軍事應用場景。

**方法說明**：

- **類別權重調整**：在模型訓練的損失函數中，給予少數類別（故障）更高的權重，讓模型「更害怕」漏判故障案例，代價是可能增加誤判（假警報）。
- **SMOTE（Synthetic Minority Over-sampling Technique）**：不是單純複製少數類別樣本，而是在特徵空間中，於少數類別樣本之間**合成新的、介於兩者之間的人工樣本**，增加少數類別的訓練樣本數量，緩解類別不平衡。
- **評估重點**：不看準確率（如 1.8 節警告，具誤導性），聚焦比較召回率（是否抓到更多真實故障）與精確率（假警報是否可接受）之間的取捨。

**模擬資料**：模擬 2000 件裝備，僅 8% 發生嚴重故障（貼近真實故障率遠低於 50% 的情境）。

```python
# ============================================================
# 論文方向二：不平衡資料分類技術比較（SMOTE / 類別權重）
# ============================================================
!pip install imbalanced-learn --quiet

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score,
                               recall_score, f1_score, roc_auc_score)
from imblearn.over_sampling import SMOTE

np.random.seed(42)
n = 2000

mileage = np.random.normal(45000, 15000, n).clip(500, None)
engine_temp = np.random.normal(88, 6, n)
vibration = np.random.gamma(2.0, 1.5, n)

# 刻意設計稀有事件：僅約8%的裝備發生嚴重故障
risk_score = 0.00003*mileage + 0.04*engine_temp + 0.2*vibration + np.random.normal(0, 0.3, n)
threshold = np.percentile(risk_score, 92)
failure = (risk_score > threshold).astype(int)
print(f"故障比例: {failure.mean()*100:.2f}%")

X = pd.DataFrame({'mileage': mileage, 'engine_temp': engine_temp, 'vibration': vibration})
y = failure
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

def evaluate(model, Xte, yte, label):
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    print(f"\n{label}:")
    print(f"  準確率={accuracy_score(yte,pred):.4f}, 精確率={precision_score(yte,pred,zero_division=0):.4f}, "
          f"召回率={recall_score(yte,pred):.4f}, F1={f1_score(yte,pred):.4f}, AUC={roc_auc_score(yte,proba):.4f}")
    print(f"  混淆矩陣:\n{confusion_matrix(yte,pred)}")

# --- 方法A：基準模型（未處理不平衡）---
clf_base = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
evaluate(clf_base, Xte, yte, "方法A：基準邏輯斯迴歸（未處理不平衡）")

# --- 方法B：類別權重調整 ---
clf_weighted = LogisticRegression(max_iter=1000, class_weight='balanced').fit(Xtr, ytr)
evaluate(clf_weighted, Xte, yte, "方法B：類別權重調整（class_weight='balanced'）")

# --- 方法C：SMOTE過採樣 ---
smote = SMOTE(random_state=42)
Xtr_smote, ytr_smote = smote.fit_resample(Xtr, ytr)
print(f"\nSMOTE前訓練集分布: {dict(pd.Series(ytr).value_counts())}")
print(f"SMOTE後訓練集分布: {dict(pd.Series(ytr_smote).value_counts())}")
clf_smote = LogisticRegression(max_iter=1000).fit(Xtr_smote, ytr_smote)
evaluate(clf_smote, Xte, yte, "方法C：SMOTE過採樣")
```

**預期輸出**：方法 A（基準模型）的召回率僅約 65%（漏判超過三分之一的真實故障案例），但精確率較高（約 79%）；方法 B（類別權重調整）與方法 C（SMOTE）的召回率皆大幅提升至約 96%（幾乎抓到所有真實故障），但精確率下降至約 44–46%（假警報增加）。**三種方法的 AUC 幾乎相同（約 0.975），代表模型的整體區辨能力並未改變，改變的只是「分類門檻決策」的取捨位置**。

> [!IMPORTANT]
> 這是不平衡資料分類研究最重要的洞察：**AUC 衡量的是模型「排序」故障風險的能力，不會因為處理不平衡的手法而改變；但「精確率／召回率」的具體取捨點，才是實務決策真正在意的**。在軍事後勤情境下，漏判一件真正會故障的關鍵裝備的代價（可能導致任務失敗），通常遠高於一次不必要的預防性檢修（假警報的代價），因此方法 B、C 這種「犧牲精確率、換取召回率」的策略，往往才是正確的決策方向——**選擇哪種方法，最終取決於漏判與誤判的相對代價，而非單純看哪個指標數字比較漂亮**。

**論文延伸建議**：可進一步建立漏判成本與誤判成本的量化函數（呼應第 5 週成本效益折衷分析的邏輯），求解使總預期成本最小的最適分類門檻，而非機械式套用 0.5；也可以比較 SMOTE 的多種變形方法（如 Borderline-SMOTE、ADASYN）在此資料上的表現差異，探討何種過採樣策略最適合軍用裝備故障預測的資料特性。

---

## 附錄A：理論常見問答（概念釐清 Q&A）

> [!NOTE]
> **Q1： $R^2$ 越高，代表模型一定越好嗎？**
> A：不一定。 $R^2$ 只衡量「配適度」，不代表模型具有因果解釋力或預測其他資料的能力。加入越多自變數， $R^2$ 必然不會下降（即使新變數毫無意義），這正是為什麼比較不同自變數個數的模型時，應該使用**調整後 $R^2$**（會對自變數個數進行懲罰）而非原始 $R^2$ 。此外， $R^2$ 很高也可能是「過度配適（Overfitting）」的警訊，須以訓練集以外的測試集驗證才能確認。

> [!NOTE]
> **Q2：線性迴歸與邏輯斯迴歸都稱為「迴歸」，兩者關係是什麼？**
> A：兩者都屬於**廣義線性模型（Generalized Linear Model, GLM）**家族——GLM 的核心概念是「自變數的線性組合，透過一個連結函數（Link Function），對應到應變數的期望值」。線性迴歸的連結函數是「恆等函數」（直接對應）；邏輯斯迴歸的連結函數是「Logit 函數」（對應到對數勝算）。理解這個共通架構，有助於未來學習其他 GLM 延伸模型（如卜瓦松迴歸，適用於計數型應變數）。

> [!NOTE]
> **Q3：勝算比等於 2，代表「機率」變成原本的 2 倍嗎？**
> A：不是，這是最常見的誤解。勝算比是「勝算（Odds）」的倍數關係，不是「機率」的倍數關係。舉例：若原本機率為 0.5（勝算為 1），勝算比 2 會讓勝算變為 2（機率變為 $2/(1+2)=0.667$ ，只增加 33%，並非變成 2 倍的機率 1.0）；但若原本機率很低（如 0.05，勝算約 0.0526），勝算比 2 會讓機率變為約 0.0952，此時機率確實幾乎變成 2 倍。**勝算比對機率的實際影響程度，會隨基準機率而變化**，這是解讀邏輯斯迴歸結果時必須謹慎之處。

> [!NOTE]
> **Q4：什麼時候該用 Ridge，什麼時候該用 Lasso？**
> A：若研究目的是**保留所有變數、只求穩定的係數估計**（如政策分析中每個變數都有理論上的重要性，不希望被排除），選擇 Ridge；若研究目的是**同時進行變數篩選**（如在大量候選預測變數中，希望模型自動找出真正重要的少數變數），選擇 Lasso。實務上也可以用 Elastic Net（同時結合 L1 與 L2 懲罰）取兩者之長，本週未詳述，有興趣的學員可自行查閱進階資料。

> [!NOTE]
> **Q5：類別權重調整與 SMOTE，兩種不平衡資料處理技術該如何選擇？**
> A：類別權重調整不會改變訓練資料本身，只調整損失函數的計算方式，計算成本低、實作簡單；SMOTE 會實際合成新的訓練樣本，可能更有效地讓模型學到少數類別的特徵空間結構，但計算成本較高、且在特徵空間結構複雜時，合成樣本可能不夠真實。實務上建議兩者都嘗試，並以驗證集的召回率／精確率表現決定採用何者，如本週 3.5 節示範的比較方法。

---

## 附錄B：本週方法快速索引表

| 任務 | 主要函數／類別 | 所屬套件 |
|---|---|---|
| 線性迴歸（含完整統計檢定） | `sm.OLS` | `statsmodels.api` |
| 邏輯斯迴歸（scikit-learn風格） | `LogisticRegression` | `sklearn.linear_model` |
| 邏輯斯迴歸（含完整統計檢定） | `sm.Logit` | `statsmodels.api` |
| VIF共線性檢查 | `variance_inflation_factor` | `statsmodels.stats.outliers_influence` |
| 異質變異性檢定 | `het_breuschpagan` | `statsmodels.stats.diagnostic` |
| 混淆矩陣與分類指標 | `confusion_matrix`、`precision_score`等 | `sklearn.metrics` |
| ROC曲線與AUC | `roc_curve`、`roc_auc_score` | `sklearn.metrics` |
| Ridge／Lasso正則化迴歸 | `Ridge`、`Lasso` | `sklearn.linear_model` |
| SMOTE過採樣 | `SMOTE` | `imblearn.over_sampling` |

---

## 附錄C：常見易混淆概念澄清

| 容易混淆的概念 | 差異說明 |
|---|---|
| 「 $R^2$ 」 vs 「調整後 $R^2$ 」 | 前者加入任何變數都不會下降；後者會對自變數個數懲罰，比較不同模型時應優先參考後者 |
| 「勝算 Odds」 vs 「機率 Probability」 | 機率是發生次數占總數比例；勝算是發生次數與不發生次數的比值，兩者數值不同、換算關係非線性 |
| 「精確率」 vs 「召回率」 | 精確率關心「預測為正的樣本中，真正為正的比例」；召回率關心「真正為正的樣本中，被正確抓到的比例」，兩者經常互相取捨 |
| 「Ridge」 vs 「Lasso」 | Ridge（L2）讓係數趨近零但不會恰好為零；Lasso（L1）會讓不重要的係數恰好壓縮為零，具備自動變數選擇效果 |
| 「類別權重調整」 vs 「SMOTE」 | 前者調整損失函數的權重，不改變訓練資料本身；後者實際合成新的少數類別樣本，改變訓練資料的組成 |

---

## 附錄D：本週與後續課程週次的關聯

| 後續週次 | 關聯方式 |
|---|---|
| 第 9 週：資料前處理 | VIF 共線性檢查、訓練/測試集切分原則，皆為本週迴歸與分類建模的前置基礎 |
| 第 11 週：非監督學習 | 本週監督式學習（有明確目標變數）與第 11 週非監督式學習（無目標變數）形成機器學習兩大典範的對照 |
| 第 12 週：決策樹與隨機森林 | 本週線性模型與第 12 週樹狀模型，是處理相同預測問題的兩種不同方法論取徑，可相互比較優劣 |
| 第 15、16 週：論文整併實戰 | 本週建立的迴歸／分類建模流程，將作為後續整合型專論中「預測模組」的核心方法 |

## 附錄E：本週模擬資料集與程式碼彙整

| 資料集 | 用途 | 摘要 |
|---|---|---|
| 簡單迴歸手算範例（10筆） | Hour1理論範例 | 里程數 vs 維修成本， $R^2=0.9952$ |
| 邏輯斯迴歸手算範例（18筆） | Hour1理論範例 | 震動值 vs 是否需要維護，勝算比=8.19 |
| 車輛感測資料（1,000筆，延續第9週） | Hour2主範例 | 新增maintenance_cost連續變數， $R^2\approx0.53$ |
| 彈藥庫儲存資料（200筆） | 延伸練習一 | 三變數多元迴歸， $R^2\approx0.48$ |
| 通信裝備故障資料（300筆） | 延伸練習二 | 二變數邏輯斯迴歸，AUC約0.82 |
| 高共線性裝備資料（300筆） | 論文方向一 | mileage/operating_hours相關係數0.99，VIF>50 |
| 稀有故障資料（2,000筆） | 論文方向二 | 故障率8%，三方法召回率/精確率比較 |

---

## 參考資料

- [迴歸分析](https://zh.wikipedia.org/wiki/%E8%BF%B4%E6%AD%B8%E5%88%86%E6%9E%90)（維基百科）
- [簡單線性迴歸](https://zh.wikipedia.org/zh-tw/%E7%B0%A1%E5%96%AE%E7%B7%9A%E6%80%A7%E8%BF%B4%E6%AD%B8)（維基百科）
- [線性迴歸](https://zh.wikipedia.org/zh-tw/%E7%B7%9A%E6%80%A7%E5%9B%9E%E6%AD%B8)（維基百科）
- [決定係數](https://zh.wikipedia.org/zh-tw/%E5%86%B3%E5%AE%9A%E7%B3%BB%E6%95%B0)（維基百科）
- [高斯-馬可夫定理](https://zh.wikipedia.org/zh-tw/%E9%AB%98%E6%96%AF-%E9%A6%AC%E5%8F%AF%E5%A4%AB%E5%AE%9A%E7%90%86)（維基百科）
- [邏輯迴歸](https://zh.wikipedia.org/zh-tw/%E9%82%8F%E8%BC%AF%E8%BF%B4%E6%AD%B8)（維基百科）
- [廣義線性模型](https://zh.wikipedia.org/zh-tw/%E5%BB%A3%E7%BE%A9%E7%B7%9A%E6%80%A7%E6%A8%A1%E5%9E%8B)（維基百科）
- [混淆矩陣](https://zh.wikipedia.org/zh-tw/%E6%B7%B7%E6%B7%86%E7%9F%A9%E9%99%A3)（維基百科）
- [靈敏度和特異度](https://zh.wikipedia.org/zh-tw/%E9%9D%88%E6%95%8F%E5%BA%A6%E5%92%8C%E7%89%B9%E7%95%B0%E5%BA%A6)（維基百科）
- [ROC曲線](https://zh.wikipedia.org/wiki/ROC%E6%9B%B2%E7%BA%BF)（維基百科）
- [嶺回歸](https://zh.wikipedia.org/zh-tw/%E5%B2%AD%E5%9B%9E%E5%BD%92)（維基百科；Ridge迴歸）

---

*下週課程：第 11 週｜非監督學習－分群分析與降維技術（Clustering & Dimensionality Reduction）。*
