# Bài 1: LoRA là gì? Cơ chế low-rank adaptation giúp fine-tune tiết kiệm
## 1.1. Giải thích cấu hình LoRA
+ `r`: hạng của cặp ma trận LoRA, có thể 16, 32 nếu các tác vụ phức tạp
+ `lora_alpha`: thường bằng $\alpha=2r$ do quy ước phổ biến $\frac{\alpha}{r}=2$
+ `lora_dropout`: bỏ ngẫu nhiên 1 số lượng phần trăm để giảm overfiting trên dataset nhỏ
+ `task_type`:
    + `SEQ_CLS`: text classification
    + `SEQ_2_SEQ_LM`: Sequence-to-sequence language modeling.
    + `CAUSAL_LM`: Causal language modeling. (decoder only)
    + `TOKEN_CLS`: Token classification.
    + `QUESTION_ANS`: Question answering.
    + `FEATURE_EXTRACTION`: Feature extraction. Provides the hidden states which can be used as embeddings or features for downstream tasks.

Nhưng cần hiểu chính xác một điều dễ nhầm: model gốc vẫn phải nằm trong VRAM ở dạng đầy đủ (vd 14 GB cho 7B ở 16-bit) để tính forward pass. LoRA chỉ cắt phần optimizer + gradient của trọng số gốc, KHÔNG làm nhỏ bản thân model gốc. Nếu nạp model gốc cũng đã vượt VRAM, bạn cần thêm mẹo lượng tử hóa 4-bit - chính là QLoRA, chủ đề bài kế tiếp: LoRA lo chi phí train, QLoRA lo nốt chi phí nạp model.

# Bài 2: QLoRA: fine-tune trên GPU nhỏ nhờ lượng tử hóa 4-bit
## 2.1. はがき
LoRA thường vẫn phải nạp model gốc ở 16-bit vào VRAM - với model 7B riêng cái đó đã ngốn ~14 GB, chưa kể optimizer và activation. GPU 8 GB của bạn chào thua.

QLoRA (Quantized LoRA) vá đúng lỗ hổng này: nó nén model gốc xuống còn 4-bit để nhét vừa GPU nhỏ, rồi vẫn gắn adapter LoRA lên trên để fine-tune. Kết quả: bạn fine-tune được một model 7B ngay trên GPU consumer (card 12-16 GB, thậm chí Colab free), điều mà trước đây phải thuê cụm GPU đắt đỏ mới làm nổi.

Một câu để nhớ: QLoRA = đóng băng + nén model gốc xuống 4-bit (để vừa VRAM) rồi train adapter LoRA nhẹ ở 16-bit trên đó - fine-tune model lớn trên GPU bé mà chất lượng gần như không giảm.

Nén 4-bit KHÔNG phải “cắt bớt não model”. Trọng số vẫn đủ, chỉ được biểu diễn thô hơn; cái hay của QLoRA là phần thô đó được bù lại bằng adapter LoRA train ở 16-bit.

## 2.2. Cốt lõi của QLoRA
|Kỹ thuật|Giải thích|Lợi ích|
|-|-|-|
|NF4 (Normal float 4-bit)|Kiểu dữ liệu 4 bit được thiết kế riêng cho trọng số có phân bố chuẩn|Nén 4 bit mà sai số nhỏ hơn `int4` thông thường|
|Double quantization| Lượng tử hóa cả hằng số lượng tử thêm 1 lần nữa|Tiết kiệm được khoảng 0.4bit/tham số|
|Paged optimizer|Dùng bộ nhớ hợp nhất (unified memory) của NVIDIA để “phân trang” state của optimizer khi VRAM sắp tràn|Chống lỗi out-of-memory lúc gặp batch dài đột biến|

```
   Model gốc 7B (16-bit, ~14 GB)
            |
            v   NÉN xuống NF4 (double quantization)
   Model gốc NF4 4-bit (~3.5 GB, ĐÓNG BĂNG - không train)
            |
            +--- gắn adapter LoRA (B, A) ở 16-bit  <- CHỈ train phần này
            |
            v
   Lan truyền ngược (backprop) chảy QUA model 4-bit,
   nhưng CHỈ cập nhật trọng số của adapter LoRA
```

## 2.3. Quy trình
```
BitsAndBytesConfig 4-bit 
          |
          v
from_pretrained nạp model 4-bit
          |
          v
prepare_model_for_kbit_training
          |
          v
LoraConfig + get_peft_model gắn adapter
          |
          v
Trainer/SFTTrainer.train() chỉ train adapter
          |
          v
save_pretrained lưu adapter nhẹ.
```
`prepare_model_for_kbit_training(model)`: bước không được quên. Nó ép layer norm về fp32 cho ổn định số, bật gradient checkpointing để tiết kiệm VRAM, và cho phép gradient chảy tới adapter. Quên bước này -> loss thường không giảm hoặc lỗi.

 Sau train, save_pretrained chỉ lưu adapter (thường vài chục MB), KHÔNG lưu lại cả model 7B. Khi dùng lại, bạn nạp model gốc rồi “dán” adapter lên. Chuyện merge adapter vào model và chạy inference sẽ học kỹ ở Bài 3 và các bài serving sau.

# Unsloth
