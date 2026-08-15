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
