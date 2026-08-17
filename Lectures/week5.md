# Bài 1: Đánh giá mô hình sau fine-tune: benchmark và LLM-as-judge
## 1.1. Benchmark tự động và LLM-as-judge
|Tiêu chí|Benchmark tự động|LLM-as-judge|
|-|-|-|
|Loại tác vụ|Có đáp án|Nhiều câu trả lời đúng|
|Cách đánh giá|So khớp hoặc tính accuracy|Dùng 1 LLM mạnh hơn để chấm theo rubric|
|Tính khách quan trong quy trình đánh giá|Có thể có bias của LLM judge|
|Ví dụ|MMLU, GSM8K, HellaSwag|Chấm chatbot|

+ `Rubric`: Rubric là một công cụ đánh giá chỉ rõ các tiêu chí đạt được trên tất cả các nhiệm vụ của sinh viên, từ nhiệm vụ bằng văn bản đến nhiệm vụ bằng miệng và nhiệm vụ trực quan. Nó có thể được sử dụng để chấm điểm đối với bài tập, sự tham gia lớp học hoặc điểm tổng kết. Có hai loại đánh giá theo tiêu chí: tổng thể và chi tiết.

## 1.2. Benchmark tự động với lm-evaluation-harness
```
# Install the harness from the official EleutherAI repo
pip install lm-eval

# Evaluate a local/HF model on MMLU (5-shot). --model hf = HuggingFace backend
lm_eval --model hf \
  --model_args pretrained=./my-merged-model,dtype=bfloat16 \ 
  --tasks mmlu \
  --num_fewshot 5 \
  --batch_size 8 \
  --output_path results/mmlu_finetuned.json
```
Với:
+ `--model_args pretrained=...`: trỏ tới thư mục model đã merge của bạn (hoặc repo trên Hub)
+ `--tasks`: chọn bộ test. Có thể liệt kê nhiều bộ test bằng cách phân tách bằng dấu `,` `mmlu,gsm8k,hellaswag`
+ `--num_fewshot`: số lượng mẫu ví dụ mà model được xem

Các bộ test:
+ `mmlu`: 
    + Đánh giá kiến thức tổng quan và suy luận trên nhiều lĩnh vực
    + Cấu trúc là dạng trắc nghiệm từ nhiều chủ đề như toán, sử, ...
    + Độ khó: trung học đến chuyên gia
    + Là một trong những benchmark "must-have" khi so sánh các mô hình lớn (GPT, Claude, Gemini, Llama...).
+ `GSM8K` (Grade School Math 8K):
    + Đánh giá khả năng suy luận toán học nhiều bước
    + Các bài toán tiểu học/THCS cần > 2 bước tính toán
    + Cách chấm: Accuracy trên đáp số cuối cùng, thường dùng kỹ thuật "chain-of-thought" (giải thích từng bước) để cải thiện kết quả.
+ `hellaswag`:
    + Đánh giá khả năng suy luận thông thường và hiểu ngữ cảnh
    + Cấu trúc: Cho một đoạn mô tả tình huống, mô hình phải chọn ra câu kết thúc hợp lý nhất trong 4 lựa chọn.
    + Điểm đặc biệt: Các câu nhiễu (distractors) được tạo bằng phương pháp "Adversarial Filtering" — trông có vẻ hợp lý về mặt cú pháp nhưng vô lý về logic thực tế, khiến các mô hình yếu dễ bị đánh lừa dù người thường thấy rất dễ.
    + Cách chấm: Accuracy trong việc chọn đúng câu kết thúc.