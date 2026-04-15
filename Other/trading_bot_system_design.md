mô hình ML

- Thu thập dữ liệu từ Vnstock (done)
- làm sạch dữ liệu (done)
- Bộ dữ liệu sử dụng là bộ dữ liệu một ngày 
- Có một điều quan trọng là nếu như mô hình sử dụng bộ dữ liệu là 1D thì liệu những biến động trong ngày thì giá nó đã chạm đến cái giá mà dự đoán hay không, nên khi training mô hình thì bot này cũng phải đưa ra được khoảng giá tiếp theo có thể vào lệnh, và khoảng giá tiếp theo có thể bán lệnh 
- mô hình sử dụng MACrossover này sẽ được tính toán dựa trên giá đóng cửa, vậy giá cao nhất, giá thấp nhất, giá mở cửa hay volume có ảnh hưởng điều gì  
- có thể MA10 cắt qua MA50 rồi nhưng volume thấp thì có thể không nên vào vì đó là fake breakout vì lực mua yếu chưa chắn là giá sẽ tăng, mô hình có thể cần phân biệt được fake breakout và breakout yếu hay breakout thật khi training model đây chính là ứng dụng của open, high, low, volume
- thiết kế mô hình, có thể bằng random forest, ...
- training mô hình, sau đó back test dữ liệu, để kiểm tra xem mô hình có hoạt động đúng hay không. 
- còn một điều nữa bởi vì giá của các của phiếu là như nhau cũng như volume là khác nhau liệu có cần chuẩn hóa chuẩn hóa trước khi đưa vào mô hình để train hay không
- có thể đưa dữ liệu vào sql (chỉ cần lưu vào 1 bảng thôi) để store rồi khi cần sẽ lấy ra để thống nhất cũng như chuẩn hóa rồi đưa vào mô hình để train được không
- sau khi kiểm tra back test mô hình sẽ in ra một sách các lệnh đã được vào theo ohlcv tại thời điểm đó rồi lưu vào file excel một cách tự động, in thêm cả ngày giờ vào lệnh theo giờ VN, bên cạnh đó tôi mong muốn mô hình cần tính toán % thắng tại điểm vào lệnh (thắng là khi tính toán được số tiền lãi thấp nhất - tiền phí giao dịch của sàn > 0) và tất cả điều này cần được training từ trước. 
- sau khi backtest xong nó sẽ được in ra tổng lãi lỗ theo % và tiền VND, số lần vào và vẽ lại mô hình các đường MA10 và MA50 
- nhưng có một điểm cần thắc mắc là mô hình vẽ theo đường MA này sẽ hầu nhưu là sẽ lãi chỉ là phụ thuộc vào thời gian mà thôi bởi vì nó sẽ mua ngay sau khi MA10 cắt lên trên MA50 nhưng nó cũng sẽ bán khi đường MA10 cắt xuống đường MA50, mô hình sẽ cần phải đường train làm sao để không phải lúc nào cắt lên cũng mua hay lúc nào xuống cũng bán, học thật kĩ khi nào nên mua và khi nào nên bán, để tránh việc có thể bị LỖ VÌ PHÍ GIAO DỊCH SỬ DỤNG SẢN KHI MUA MỘT CỔ PHIẾU, sau khi back test xong còn phải in ra Classification report nữa  
- về phần UI, có cần một UI nào để có thể giao tiếp được với bot hay không, như kích hoạt để sử dụng bot cũng như tắt kích hoạt bot, tick vào những cổ phiếu mà cho phép bot giao dịch, quan trọng hơn cả là bây giờ có sàn nào để cho phép lấy API để trade bằng terminal (tôi đã xem và có SSI đã được có lẽ tôi sẽ chọn SSI để tự động giao dịch)
- VÀ ĐIỀU QUAN TRỌNG LÀ PHẢI ỨNG DỤNG ĐƯỢC VÀO THỰC TẾ 


NOTE: thêm vào đó mô hình UI để tương tác với bot, thì mỗi một mã cổ phiếu sẽ chỉ được mua 1 lần duy nhất (tùy số lượng cổ phiếu) để đảm bảo Risk Management, có thể mua nhiều mã cổ phiếu cùng 1 lúc


Link một vài DIAGRAM về quy trình của project: https://drive.google.com/drive/folders/17YiWbPSQf12buyKTuSlyT8V7hwJ6tcoc?usp=drive_link

<!-- MONGO_URI = "mongodb+srv://doankhangll255_db_user:EGyi6XqdcCAwxbrf@cluster0.ufdio5k.mongodb.net/?retryWrites=true&w=majority" -->
