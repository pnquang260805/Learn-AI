# Bài 1: Các loại fine-tuning: full fine-tune, PEFT, instruction tuning, alignment
## 1.1. PEFT và LoRA
+ `PEFT (Parameter-EfFicient Fine-Tuning)` là kĩ thuật chỉ huấn luyện 1 phần rất nhỏ các tham số, `đóng băng` gần như toàn bộ trọng số gốc
+ `LoRA` là kĩ thuật nổi tiếng nhất của `PEFT`

$W' = W + \Delta W = W + BA$

Với:
+ $B$ và $A$ và 2 ma trận phân rã
+ Nếu $W$ có kích thước $d \times k$ thì `B` là $d \times r$ và `A` có kích thước là $r \times k$ với `rank` $r$ rất nhỏ (thường 8, 16, 32) và $r<<min(d, k)$
+ Số tham số train giảm cả trăm lần

Cơ chế LoRA (chỉ B và A được huấn luyện):

```mermaid
graph TD
    X[đầu vào x] --> W["[ W ] (đóng băng - không train)"]
    X --> A["[ A ]"]
    A --> B["[ B ] (hạng thấp r, ĐƯỢC train)"]
    
    W --> Plus(( + ))
    B --> Plus
    
    Plus --> Y["cộng lại: y = W·x + B·A·x"]
```

+ Chúng ta tối ưu $\Delta W$ bởi $\frac{\alpha}{r}$ ($\alpha$ là biến `lora_alpha` trong `LoraConfig` đóng vai trò như `learning rate`)
+ `target_modules` là các lớp được gắn adapter (theo __paper__ thì gắn vào vector $Q$ và vector $V$ cho ra kết quả tốt nhất)

+ QLoRA: nạp model gốc đã được lượng tử hóa rồi mới gắn LoRA lên trên

## 1.2. Instruction tuning (SFT): dạy model nghe lệnh làm đúng nhiệm vụ
+ Model pre-train chỉ có thể đoán được từ tiếp theo chứ chưa quan nghe lệnh (Ví dụ: "Hãy viết 1 email")
+ `Instruction tuning` (SFT - Supervised Fine-Tuning): huấn luyện model trên các cặp `(chỉ dẫn, câu trả lời)

## 1.3. Alignment (DPO/RLHF): dạy model các cư xử như người
+ SFT: chưa đảm bảo model cho ra output ưng ý con người dù output đấy đúng
+ RLHF (Reinforcement Learning from Human Feedback): cần train thêm reward model
+ DPO (Direct Preference Optimization): học trực tiếp từ (câu ưa thích, câu bị chê) và không cần reward model

## 1.4. Note
❌ “Làm DPO thì bỏ qua SFT cho nhanh.” → ✅ Sai thứ tự. Nên SFT trước rồi mới alignment.

# Bài 2: Chuẩn bị dữ liệu huấn luyện: định dạng instruction và chat
Tưởng tượng bạn thuê một gia sư siêu giỏi (model nền đã pretrain) về dạy kèm. Gia sư này biết cực nhiều, nhưng nói lan man, hỏi một trả lời mười, đôi khi lạc đề. Việc của bạn là đưa cho gia sư một cuốn sổ tay ghi rõ: “Khi học sinh hỏi kiểu này, hãy trả lời kiểu này”. Cuốn sổ tay đó chính là dataset huấn luyện. Fine-tune giỏi hay dở, 80% nằm ở chất lượng cuốn sổ này, chứ không phải ở thuật toán.

## 2.1. Chat template
Mỗi model có 1 chat template riêng. Ví dụ:

   ChatML (Qwen va nhieu model open):<br>
   <|im_start|>system ... <|im_end|><br>
   <|im_start|>user ... <|im_end|><br>
   <|im_start|>assistant ... <|im_end|><br>

   Llama 3 (định dạng riêng của Meta):<br>
   <|begin_of_text|><|start_header_id|>system<|end_header_id|> ... <|eot_id|><br>
   <|start_header_id|>user<|end_header_id|> ... <|eot_id|><br>

Thay vì tự ghép bằng tay thì ta để `tokenizer` tự lo bằng cách sử dụng `apply_chat_template()`

## 2.2. Loss masking: chỉ chấm bài trên phần assistant nói
SFTTrainer hỗ trợ điều này qua tham số assistant_only_loss (khi dataset ở định dạng chat và chat template có đánh dấu vùng assistant). Bật nó lên giúp model tập trung học cách trả lời, không phí công học lại câu hỏi.

# Bài 3: Chất lượng dữ liệu
## 3.1. Ví dụ về clean data
Lấy content trong các tag của HTML $\rightarrow$ lọc bỏ các mẫu rỗng, quá ngắn (thường là vô nghĩa) hoặc quá dài (tốn token, vượt context)

## 3.2. Khử trùng lặp
Có 2 loại dedup:
+ Exact dedup: hai mẫu có kí tự giống hệt nhau
+ Near-dedup: dựa trên similarity

### Near-dedup
Ở đây sử dụng `MinHash`

Thuật toán `MinHash`:
+ Dùng để ước lượng nhanh độ tương tự Jaccard giữa 2 tập hợp lớn mà không cần so sánh trực tiếp
+ Jaccard: $J(A,B)=\frac{|A\cap B|}{|A\cup B|}$
+ Hàm băm $H(x)=ax+b$ với $a, b$ là 2 tham số ngẫu nhiên thay đổi qua từng lần chạy

Quy trình:
1. Tạo tập hợp từ văn bản (Shingling): tách văn bản thành tập hợp với các thành phần có $k$ phần tử. Ví dụ câu: `con mèo trèo cây cau` với $k=2$ ta sẽ có: `{con mèo, mèo trèo, trèo cây, cây cau}`. Hàm băm: $h_{min}(A)=\min_{x\in A}h_i(x)$. Tính chất: $P(h_{min}(A)=h_{min}(B))=J(A,B)$
2. Tạo ma trận đặc trưng: là ma trận có kiểu như hình dưới

<img src="../images/shingles.png">
3. ???????????????????????????????????


### Note
Với các dataset nhỏ (dưới vài nghìn dòng) thì chỉ cần exact dedup là đủ

## 3.3. Quy trình chung cho xử lý dữ liệu huấn luyện
```mermaid
flowchart TD
    A[Dữ liệu thô<br>raw.jsonl] --> B[Làm sạch]
    B -->|Bỏ rác HTML, chuẩn hóa khoảng trắng| C[Lọc]
    C -->|Bỏ mẫu rỗng / quá ngắn / quá dài| D[Khử trùng lặp]
    D -->|Exact + near-dup MinHash| E[Chia tách]
    E -->|train_test_split<br>test_size=0.1, seed=42| F[train.jsonl]
    E -->|train_test_split<br>test_size=0.1, seed=42| G[val.jsonl]

    F & G --> H[Nạp vào SFTTrainer / Đánh giá]
```

|Tỉ lệ split thường dùng|Use case|
|-----------------------|--------|
|90-10|Mặc định phổ biến cho fine-tune LLM|
|95-5|Dataset lớn, muốn tận dụng tối đa cho train|
|80-20|Dataset nhỏ, cần validation đủ mẫu để tin cậy|

Luôn đặt `seed` cố định để mỗi lần chạy chia y hệt nhau - điều kiện tiên quyết để so sánh công bằng (reproducibility).

SAI:  raw -> split -> dedup(train), dedup(val) (bản trùng giữa train và val vẫn lọt -> LEAKAGE)

ĐÚNG: raw -> clean -> DEDUP toàn bộ -> SPLIT (đã bỏ trùng trên toàn tập trước khi tách)
