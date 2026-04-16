---
title: "Model Terms and Features Guide"
output: html_document
---

# Mục tiêu của tài liệu này

File này giải thích các thuật ngữ tài chính và các feature đang được dùng trong
`training_RF_demo_core_.py`. Mục tiêu là giúp bạn:

- hiểu ý nghĩa tài chính của từng biến,
- hiểu công thức tính,
- hiểu model đang "nhìn" thị trường qua lăng kính nào.

Tài liệu này đi theo đúng logic của project hiện tại:

- dữ liệu là OHLCV theo ngày (`1D`),
- tín hiệu nền là `MA10` cắt lên `MA50`,
- model học để lọc tín hiệu tốt/xấu,
- label được tạo bằng một trade giả định: vào ở `open` ngày hôm sau, có
  `target`, `stop`, `timeout`, và `fee`.

# 1. Nền tảng dữ liệu giá

## 1.1 OHLCV là gì

OHLCV là bộ dữ liệu chuẩn của nến giá:

- `Open`: giá mở cửa
- `High`: giá cao nhất trong phiên
- `Low`: giá thấp nhất trong phiên
- `Close`: giá đóng cửa
- `Volume`: khối lượng giao dịch

Trong project này, mỗi dòng dữ liệu là một cây nến ngày.

## 1.2 Timeframe là gì

`Timeframe` là độ dài của một cây nến. Ở đây:

- `1D` nghĩa là 1 cây nến = 1 ngày giao dịch

Điều này rất quan trọng vì với dữ liệu ngày, bạn chỉ biết:

- hôm đó giá cao nhất là bao nhiêu,
- thấp nhất là bao nhiêu,

nhưng bạn **không biết trong ngày giá chạm cái nào trước**. Vì vậy project phải
có `ambiguity_mode` để xử lý trường hợp cùng một ngày vừa chạm target vừa chạm
stop.

# 2. Thuật ngữ giá và nến

## 2.1 Candle range

Biên độ toàn cây nến:

$$
\text{Candle Range} = High - Low
$$

Biên độ càng lớn, phiên đó càng biến động mạnh.

## 2.2 Body

Thân nến là chênh lệch giữa giá mở và giá đóng:

$$
\text{Body} = |Close - Open|
$$

Nếu thân nến lớn, thị trường thường có một hướng di chuyển rõ hơn trong phiên.

## 2.3 Upper wick

Râu trên là đoạn từ phần trên thân nến đến giá cao nhất:

$$
\text{Upper Wick} = High - \max(Open, Close)
$$

Râu trên dài thường gợi ý rằng giá từng tăng cao nhưng sau đó bị bán xuống.

## 2.4 Lower wick

Râu dưới là đoạn từ giá thấp nhất đến phần dưới thân nến:

$$
\text{Lower Wick} = \min(Open, Close) - Low
$$

Râu dưới dài thường gợi ý rằng có lực mua đỡ ở vùng giá thấp.

## 2.5 Close position

Vị trí giá đóng cửa trong toàn bộ cây nến:

$$
\text{Close Position} = \frac{Close - Low}{High - Low}
$$

Giá trị gần:

- `1`: đóng cửa gần đỉnh ngày, thường mạnh
- `0`: đóng cửa gần đáy ngày, thường yếu

# 3. Thuật ngữ xu hướng

## 3.1 Moving Average (MA)

MA là trung bình động. Với MA đơn giản:

$$
MA_n(t) = \frac{Close_t + Close_{t-1} + ... + Close_{t-n+1}}{n}
$$

Trong project:

- `MA10`: trung bình đóng cửa 10 phiên
- `MA50`: trung bình đóng cửa 50 phiên

MA ngắn hạn phản ứng nhanh hơn. MA dài hạn phản ứng chậm hơn.

## 3.2 EMA

EMA là trung bình động hàm mũ, đặt trọng số cao hơn cho dữ liệu mới.

EMA thường phản ứng nhanh hơn MA đơn giản.

Trong project, EMA được dùng để tính MACD:

- `EMA12`
- `EMA26`
- `EMA9` của MACD signal

## 3.3 Golden cross

Golden cross là lúc MA ngắn hạn cắt lên MA dài hạn.

Trong project:

$$
\text{Golden Cross} =
\begin{cases}
1, & \text{nếu } MA10_{t-1} \le MA50_{t-1} \text{ và } MA10_t > MA50_t \\
0, & \text{ngược lại}
\end{cases}
$$

Đây là tín hiệu nền. Model hiện tại không tự tìm mọi điểm vào trên chart, mà
lọc các điểm đã có `golden_cross`.

## 3.4 Slope

Độ dốc ở đây được đo bằng phần trăm thay đổi trong 3 phiên:

$$
\text{MA10 Slope 3D} = \frac{MA10_t}{MA10_{t-3}} - 1
$$

$$
\text{MA50 Slope 3D} = \frac{MA50_t}{MA50_{t-3}} - 1
$$

Slope dương cho thấy đường MA đang đi lên.

# 4. Thuật ngữ momentum

## 4.1 Return

Return là mức thay đổi tương đối của giá:

$$
Return_n = \frac{Close_t}{Close_{t-n}} - 1
$$

Trong project:

- `return_3d`
- `return_5d`

Return dương lớn cho thấy đà tăng đang mạnh, nhưng đôi khi cũng có thể báo hiệu
giá đã tăng nóng.

## 4.2 RSI

RSI là chỉ báo đo độ mạnh tương đối của lực tăng so với lực giảm.

### Bước 1: tính delta

$$
\Delta_t = Close_t - Close_{t-1}
$$

### Bước 2: tách gain và loss

$$
Gain_t = \max(\Delta_t, 0)
$$

$$
Loss_t = \max(-\Delta_t, 0)
$$

### Bước 3: dùng Wilder smoothing

Giá trị đầu tiên:

$$
AvgGain_{14} = \text{mean}(Gain_1,...,Gain_{14})
$$

$$
AvgLoss_{14} = \text{mean}(Loss_1,...,Loss_{14})
$$

Từ các phiên sau:

$$
AvgGain_t = \frac{AvgGain_{t-1} \times 13 + Gain_t}{14}
$$

$$
AvgLoss_t = \frac{AvgLoss_{t-1} \times 13 + Loss_t}{14}
$$

### Bước 4: tính RS và RSI

$$
RS_t = \frac{AvgGain_t}{AvgLoss_t}
$$

$$
RSI_t = 100 - \frac{100}{1 + RS_t}
$$

Diễn giải trực giác:

- RSI cao: lực tăng gần đây mạnh hơn lực giảm
- RSI thấp: lực giảm gần đây mạnh hơn lực tăng

Quan trọng:

- RSI cao **không tự động** có nghĩa là phải giảm ngay
- RSI cao cũng có thể là xu hướng đang khỏe
- vì vậy RSI cần được đọc cùng các feature khác như volume, MACD, distance với MA

## 4.3 MACD

MACD là chênh lệch giữa EMA nhanh và EMA chậm:

$$
MACD = EMA12 - EMA26
$$

Signal line:

$$
Signal = EMA9(MACD)
$$

Histogram:

$$
MACD\ Histogram = MACD - Signal
$$

Histogram dương và tăng thường gợi ý momentum tăng đang mạnh dần.

# 5. Thuật ngữ volume

## 5.1 Volume

`Volume` là khối lượng giao dịch. Nó cho biết mức độ tham gia của dòng tiền.

Trong breakout, volume thường được dùng để kiểm tra xem cú tăng có được xác
nhận hay không.

## 5.2 Relative volume

Trong project:

$$
\text{Volume Ratio 20} = \frac{Volume_t}{MA20(Volume)_t}
$$

Nếu lớn hơn `1`, volume hôm nay lớn hơn trung bình 20 phiên.

## 5.3 Volume change

$$
\text{Volume Change Pct} = \frac{Volume_t}{Volume_{t-1}} - 1
$$

Biến này đo xem khối lượng đang tăng đột ngột hay không.

# 6. Thuật ngữ volatility

## 6.1 Volatility

Volatility là mức độ biến động giá. Giá biến động càng mạnh thì volatility càng
cao.

## 6.2 True Range

True Range là khối xây dựng nên ATR. Nó lấy giá trị lớn nhất của 3 đại lượng:

$$
TR_t = \max
\left(
High_t - Low_t,\;
|High_t - Close_{t-1}|,\;
|Low_t - Close_{t-1}|
\right)
$$

Mục đích là bắt cả:

- biên độ trong ngày,
- gap lên,
- gap xuống.

## 6.3 ATR

ATR là trung bình động của True Range. Trong project:

$$
ATR_{14}(t) = MA14(TR_t)
$$

Để so sánh giữa các mã có mức giá khác nhau, project dùng:

$$
ATR\ Pct = \frac{ATR}{Close}
$$

## 6.4 Rolling standard deviation

Trong project:

$$
\text{Rolling Std 10} = StdDev\left(Return_{1d}\right)_{10}
$$

Đây là độ lệch chuẩn của daily return trong 10 phiên gần nhất.

# 7. Thuật ngữ giao dịch và label

## 7.1 Signal

`Signal` là thời điểm mô hình xem xét có setup hay không.

Ở mode `crossover_only`, signal là các dòng có `golden_cross = 1`.

## 7.2 Entry

Project giả định:

- nhìn thấy signal ở ngày `t`
- vào lệnh ở `open` ngày `t+1`

Điều này sạch hơn so với việc giả định mua ngay ở `close` cùng ngày signal, vì
thực tế bạn chỉ biết đầy đủ cây nến ngày sau khi phiên đã kết thúc.

## 7.3 Target return

`target_return` trong project được hiểu là **mục tiêu lợi nhuận net sau phí**.

Nếu:

- `target_return = 0.03`
- `fee_pct = 0.003`

thì mức tăng gross cần đạt xấp xỉ:

$$
\text{Required Gross Return} = target\_return + fee\_pct
$$

## 7.4 Stop loss

Stop loss là mức lỗ tối đa cho phép:

$$
\text{Stop Price} = EntryPrice \times (1 + stop\_loss)
$$

Ví dụ `stop_loss = -0.02` nghĩa là cắt lỗ quanh mức `-2%`.

## 7.5 Gross return và net return

Gross return:

$$
\text{Gross Return} = \frac{ExitPrice}{EntryPrice} - 1
$$

Net return:

$$
\text{Net Return} = Gross Return - fee\_pct
$$

## 7.6 Timeout

Nếu trong `horizon` phiên:

- không chạm target,
- không chạm stop,

thì lệnh thoát ở `close` của phiên cuối cùng. Lý do thoát lúc này là
`timeout`.

## 7.7 bars_held

`bars_held` là số bar từ ngày signal đến ngày exit:

$$
bars\_held = exit\_idx - signal\_idx
$$

Vì project vào ở `t+1`, nên cột này hữu ích để so sánh tương đối thời gian giữ
lệnh, nhưng không hoàn toàn trùng với số ngày thực sự ở trong vị thế.

## 7.8 target_hit

`target_hit` là cờ nhị phân:

- `1`: trade có chạm target
- `0`: trade không chạm target

Lưu ý:

- `target_hit = 0` nhưng `buy_label = 1` vẫn có thể xảy ra
- ví dụ trade không hit TP, nhưng timeout xong vẫn lãi sau phí

## 7.9 buy_label

Hiện tại label của project là:

$$
buy\_label =
\begin{cases}
1, & \text{nếu Net Return} > 0 \\
0, & \text{ngược lại}
\end{cases}
$$

Điểm quan trọng:

- `buy_label` **không đồng nghĩa** với `target_hit`
- `buy_label = 1` nghĩa là trade giả định có lãi ròng

# 8. Giải thích từng feature trong model

## 8.1 Nhóm trend

### `ma10_ma50_ratio`

$$
\frac{MA10}{MA50}
$$

Ý nghĩa:

- > `1`: MA ngắn hạn đang nằm trên MA dài hạn
- khoảng cách vừa phải thường tốt hơn việc quá xa

### `close_ma10_ratio`

$$
\frac{Close}{MA10}
$$

Ý nghĩa:

- đo giá hiện tại cách MA10 bao xa
- nếu quá cao, có thể entry đã muộn

### `dist_ma10_ma50_pct`

$$
\frac{MA10 - MA50}{MA50}
$$

Ý nghĩa:

- đo khoảng cách tương đối giữa 2 đường MA
- giúp phân biệt "trend mới hình thành" với "trend đã kéo giãn"

### `ma10_slope_3d`

$$
\frac{MA10_t}{MA10_{t-3}} - 1
$$

Ý nghĩa:

- MA10 có đang đi lên không
- có đang tăng tốc hay không

### `ma50_slope_3d`

$$
\frac{MA50_t}{MA50_{t-3}} - 1
$$

Ý nghĩa:

- xu hướng nền dài hơn có đang ủng hộ trade không

### `golden_cross`

Biến cờ:

- `1`: hôm nay vừa xảy ra `MA10` cắt lên `MA50`
- `0`: không phải ngày crossover

Trong mode `crossover_only`, biến này thường bị loại khỏi model vì sau khi lọc
nó trở thành cột hằng số.

## 8.2 Nhóm momentum

### `return_3d`

$$
\frac{Close_t}{Close_{t-3}} - 1
$$

Ý nghĩa:

- đo đà tăng ngắn hạn
- giúp biết giá vừa tăng đủ đẹp hay đã nóng

### `return_5d`

$$
\frac{Close_t}{Close_{t-5}} - 1
$$

Ý nghĩa:

- nhìn momentum dài hơn một chút so với 3 ngày

### `rsi`

Dùng Wilder RSI 14 phiên.

Ý nghĩa:

- đo sức mạnh của lực tăng so với lực giảm
- là chỉ báo động lượng đã được chuẩn hóa về thang 0-100

### `macd_hist`

$$
MACD - Signal
$$

Ý nghĩa:

- histogram dương và tăng thường tốt cho phía mua
- histogram âm hoặc giảm cho thấy đà đang yếu đi

## 8.3 Nhóm volume

### `volume_ratio_20`

$$
\frac{Volume_t}{MA20(Volume)_t}
$$

Ý nghĩa:

- volume hôm nay có đang bất thường cao không
- breakout mạnh thường đi kèm volume xác nhận

### `volume_change_pct`

$$
\frac{Volume_t}{Volume_{t-1}} - 1
$$

Ý nghĩa:

- đo đột biến volume so với ngày trước

## 8.4 Nhóm volatility

### `atr_pct`

$$
\frac{ATR_{14}}{Close}
$$

Ý nghĩa:

- đo biến động tương đối
- giúp model phân biệt mã đang quá rung lắc với mã đang biến động vừa phải

### `rolling_std_10`

Độ lệch chuẩn của daily return trong 10 phiên gần nhất.

Ý nghĩa:

- càng cao thì giá càng biến động

## 8.5 Nhóm candle

### `body_ratio`

$$
\frac{|Close - Open|}{High - Low}
$$

Ý nghĩa:

- thân nến chiếm bao nhiêu phần trong toàn bộ cây nến
- thân lớn thường cho thấy cây nến quyết đoán hơn

### `upper_wick_ratio`

$$
\frac{UpperWick}{High - Low}
$$

Ý nghĩa:

- râu trên dài có thể là dấu hiệu bị từ chối giá

### `lower_wick_ratio`

$$
\frac{LowerWick}{High - Low}
$$

Ý nghĩa:

- râu dưới dài có thể cho thấy lực đỡ từ bên mua

### `close_position`

$$
\frac{Close - Low}{High - Low}
$$

Ý nghĩa:

- close càng gần đỉnh nến thì lực cuối phiên càng tích cực

# 9. Cách đọc bộ feature như một bức tranh thị trường

Model hiện tại không "nhìn chart" như con người. Nó nhìn qua các biến số. Có thể
hiểu từng nhóm feature như sau:

- nhóm trend: xu hướng có tồn tại không
- nhóm momentum: lực tăng có đang khỏe không
- nhóm volume: breakout có được xác nhận không
- nhóm volatility: trade có quá rung lắc không
- nhóm candle: cây nến ngay tại signal có mạnh hay yếu

Khi ghép lại, model đang cố học quy luật kiểu:

- một golden cross nào thì đáng mua,
- một golden cross nào chỉ là fake breakout hoặc cú tăng đã muộn.

# 10. Ghi chú quan trọng khi dùng tài liệu này

Project hiện tại đang dùng dữ liệu ngày, nên một số khái niệm về target/stop có
giới hạn:

- nếu cùng một ngày vừa chạm target vừa chạm stop thì không biết cái nào xảy ra
  trước,
- vì vậy kết quả label và backtest vẫn là một xấp xỉ hợp lý, chưa phải mô phỏng
  execution hoàn hảo.

Tài liệu này bám đúng logic đang chạy trong code hiện tại, không phải lý thuyết
TA chung chung.
