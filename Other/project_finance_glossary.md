# Project Finance Glossary

File này là bản glossary ngắn gọn hơn để bạn tra cứu nhanh khi đọc project.
Khác với file `.Rmd`, tài liệu này ưu tiên cách hiểu thực chiến hơn là trình bày
công thức dài.

## 1. Thuật ngữ thị trường cơ bản

### Ticker

Mã giao dịch của một tài sản, ví dụ `ACB`, `FPT`, `VCB`.

### OHLCV

Bộ 5 giá trị chuẩn của một cây nến:

- `Open`: giá mở cửa
- `High`: giá cao nhất
- `Low`: giá thấp nhất
- `Close`: giá đóng cửa
- `Volume`: khối lượng giao dịch

### Timeframe

Độ dài của một cây nến. Trong project hiện tại là `1D`, tức mỗi dòng dữ liệu là
một ngày giao dịch.

### Candle

Một cây nến đại diện cho diễn biến giá trong một khoảng thời gian.

### Bullish / Bearish

- `Bullish`: thiên tăng
- `Bearish`: thiên giảm

## 2. Thuật ngữ xu hướng

### Trend

Xu hướng chính của giá. Project đang cố phân biệt:

- có trend thật,
- hay chỉ là tín hiệu tăng giả.

### Moving Average (MA)

Trung bình động của giá. Dùng để làm mượt dữ liệu và nhìn xu hướng.

### EMA

Một dạng MA phản ứng nhanh hơn với dữ liệu mới.

### Golden Cross

MA ngắn hạn cắt lên MA dài hạn. Trong project, đó là:

- `MA10` cắt lên `MA50`

Đây là tín hiệu nền để model lọc.

### Slope

Độ dốc của MA. Dùng để kiểm tra xem xu hướng đang đi lên nhanh hay chậm.

## 3. Thuật ngữ động lượng

### Momentum

Độ mạnh của chuyển động giá. Giá có thể đang tăng, nhưng câu hỏi là:

- tăng yếu,
- hay tăng mạnh và được xác nhận.

### Return

Mức thay đổi phần trăm của giá so với quá khứ.

### RSI

Chỉ báo đo lực tăng so với lực giảm. Không nên hiểu RSI cao là "chắc chắn phải
đảo chiều". RSI cao cũng có thể là xu hướng đang mạnh.

### MACD

Chỉ báo động lượng dựa trên chênh lệch giữa EMA nhanh và EMA chậm.

### MACD Histogram

Phần chênh giữa MACD và signal line. Dùng để đo xem đà tăng/giảm đang mạnh thêm
hay yếu đi.

## 4. Thuật ngữ volume và volatility

### Volume

Khối lượng giao dịch. Tăng giá đi kèm volume tốt thường đáng tin hơn tăng giá
trong trạng thái volume yếu.

### Relative Volume

Volume hiện tại so với volume trung bình gần đây.

### Volatility

Mức độ rung lắc của giá.

### ATR

Một chỉ báo volatility phổ biến, đo biên độ biến động thực tế.

### Rolling Standard Deviation

Độ lệch chuẩn của return trong một cửa sổ thời gian. Cũng là cách đo volatility.

## 5. Thuật ngữ giao dịch trong project

### Signal date

Ngày phát hiện setup.

### Entry date

Ngày vào lệnh giả định. Project đang dùng:

- signal ở ngày `t`
- entry ở `open` ngày `t+1`

### Entry price

Giá vào lệnh giả định.

### Exit date

Ngày thoát lệnh giả định.

### Exit price

Giá thoát lệnh giả định.

### Target

Mức lợi nhuận kỳ vọng.

### Stop loss

Mức cắt lỗ.

### Timeout

Hết số phiên tối đa cho phép giữ lệnh mà vẫn chưa chạm target hoặc stop. Khi đó
trade sẽ thoát ở `close` phiên cuối.

### Horizon

Số phiên tối đa mà trade được giữ.

### Bars held

Số bar mà trade được giữ theo logic hiện tại. Trong code hiện tại, nó được tính
từ ngày signal đến ngày exit.

### Fee

Chi phí giao dịch ước lượng. Project đang giả định `0.3%`.

### Gross return

Lợi nhuận thô, chưa trừ phí.

### Net return

Lợi nhuận ròng, sau khi trừ phí.

### Target hit

Cờ cho biết trade có chạm target hay không.

### Buy label

Nhãn học máy hiện tại:

- `1`: trade giả định có lãi ròng sau phí
- `0`: trade giả định không có lãi ròng

Điều này rất quan trọng vì:

- `buy_label` không giống `target_hit`

## 6. Thuật ngữ ML và backtest trong project

### Feature

Biến đầu vào cho model. Ví dụ:

- `rsi`
- `macd_hist`
- `volume_ratio_20`

### Label

Kết quả đúng mà model cần học. Trong project này, label là `buy_label`.

### Candidate

Một dòng dữ liệu được phép đi vào model sau khi đã qua bước lọc.

### `crossover_only`

Chỉ lấy những dòng có `golden_cross = 1`.

### `all_rows`

Lấy tất cả các dòng đủ feature.

### Train set

Phần dữ liệu dùng để huấn luyện model.

### Test set

Phần dữ liệu dùng để đánh giá model trên giai đoạn thời gian chưa được dùng để
train.

### Chronological split

Chia train/test theo thời gian, không random. Đây là cách hợp lý hơn cho dữ liệu
giao dịch.

### `pred_proba`

Xác suất model dự đoán rằng trade thuộc class `1`.

### `prob_threshold`

Ngưỡng xác suất để biến dự đoán mềm thành quyết định mua:

- nếu `pred_proba >= prob_threshold` thì chọn trade

### Accuracy

Tỷ lệ dự đoán đúng trên tập test. Đây là chỉ số hữu ích, nhưng với trading thì
không quan trọng bằng PnL và win rate.

### Win rate

Tỷ lệ trade thắng trong các trade đã được chọn.

### PnL

Profit and Loss, tức lãi/lỗ.

### Backtest

Mô phỏng xem chiến lược sẽ cho kết quả ra sao nếu chạy trên dữ liệu lịch sử.

## 7. Thuật ngữ dễ gây nhầm trong project này

### Tại sao `target_hit = 0` mà `buy_label = 1` vẫn có thể đúng

Vì:

- trade không hit target,
- nhưng đến lúc `timeout` vẫn thoát ra với lợi nhuận dương sau phí.

### Tại sao accuracy cao mà PnL vẫn âm

Vì model có thể giỏi ở việc đoán lệnh xấu là xấu, nhưng chưa giỏi chọn ra nhóm
lệnh tốt để vào tiền thật.

### `ambiguity_mode`

Với dữ liệu ngày, có thể cùng một nến vừa chạm target vừa chạm stop. Khi đó:

- `conservative`: giả định stop xảy ra trước
- `optimistic`: giả định target xảy ra trước

Mode hiện tại của project là `conservative`.
