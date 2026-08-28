# Bài 1: vLLM là gì? PagedAttention và vì sao nó nhanh
## 1.1. Nhắc lại KV Cache
+ Khi LLM sinh token, nó lưu toàn bộ ngữ cảnh của các từ trước đó vào 1 vùng nhớ `KV Cache` trên GPU

## 1.2. Cách lưu trữ cũ và PagedAttention
+ Cách lưu trữ cũ (các engine như ollama, ...):
    + Cấp phát các chỗ nhớ liên tục trên VRAM như cách __cấp phát ô nhớ liên tục trong OS__
    + Gây phân mảnh nội bộ: không dùng hết các khối đã được cấp phát
    + Phân mảnh ngoại bộ: tồn tại các khối trống không đủ dung lượng để cấp phát cho bất kỳ request nào dù tổng của chúng vẫn đủ
+ PagedAttention:
    + Giống phân trang trong OS (chương trình được chia thành các khối có kích thước cố định được gọi là `page`. RAM được chia thành các khối có kích thước cố định hoặc không gọi là `frame` - xem lại file học OS. OS sẽ sử dụng `page table` để quản lý vị trí các `page` được lưu trong `frame`)
    + PagedAttention chia `KV Cache` thành các `block` cố định và chứa 1 số lượng token nhất định (ví dụ `16 token mỗi block`)
    + Các block không nhất thiết phải liền kề trong VRAM
    + Quản lý bằng Block Table: Một bảng ánh xạ giúp mô hình biết: "Logic là token 0 đến 15, nhưng trên GPU thật nó đang nằm ở Block 7".
    + Có thể cấp phát thei nhu cầu

```
   PagedAttention: KV cache = nhiều block nhỏ + block table
   Chuỗi "A" (logic):  [tok0..15][tok16..31][tok32..40]
   block table (A):        ▼          ▼          ▼
   Bộ nhớ GPU vật lý:  [Block 7] [Block 2] [Block 9]   <- rải rác, không cần liền kề
                       [Block 3=trống] [Block 5=của chuỗi B] ...
   -> Cấp block CHỈ khi cần token mới; trả lại khi chuỗi xong.
```
## 1.3. Continuous batching: không để GPU ngồi chơi
+ Giống điều độ tiến trình trong OS
+ Thường dùng với PagedAttention
+ Giải quyết phần lập lịch

Ví dụ
+ Giả sử GPU có thể xử lý tối đa 3 request cùng lúc (Batch size = 3).
+ Ban đầu nạp 3 request: A (12 token), B (5 token), C (10 token).
+ Trong hàng đợi (Ready Queue) đang có sẵn Req D (4 token) và Req E (3 token) đang chờ.

__Static batching__
<img src="../images/static batching.png"><br>
__Continuous batching__
<img src="../images/Continuous batching.png">

## 1.4. Code mẫu chạy vLLM inference offline
```python
from vllm import LLM, SamplingParams

# A small instruction-tuned model that fits on modest GPUs
llm = LLM(model="Qwen/Qwen2.5-0.5B-Instruct")

# Decoding controls: creativity (temperature) and max output length
sampling = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=128)

prompts = [
    "Explain what a KV cache is in one sentence.",
    "Vietnam's capital city is",
]

# vLLM batches these prompts internally with continuous batching
outputs = llm.generate(prompts, sampling)

for out in outputs:
    # out.prompt is the input; out.outputs[0].text is the generated text
    print(out.prompt, "->", out.outputs[0].text.strip())
```

## 1.5. Lựa chọn vLLM hay các lựa chọn khác
+ Cá nhân chạy local -> Ollama
+ Serving production trên GPU -> vLLM; nghiên cứu/prototype -> Transformers thuần. Chúng bổ trợ nhau