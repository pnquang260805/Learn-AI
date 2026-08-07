# Embeddings và xây dựng RAG cục bộ với Ollama
## Các khái niệm
|Khái niệm|Giải thích|Ví dụ|
|---------|----------|-----|
|Embedding vector|Dãy số biểu diễn nghĩa của văn bản|[0.12, -0.98, ..., 0.03] dài 768 số|
|Số chiều (dimension)|Độ dài vector|Ví dụ: 768|
|Độ tương đồng (cosine similarity)|Số đo hai vector gần nhau cỡ nào|Càng gần 1 = càng giống nghĩa|
|Không gian ngữ nghĩa|Không gian chứa mọi vector|Câu cùng chủ đề tụ lại thành cụm|

Công thức đo

$\text{cosine}(A, B) = \frac{A \cdot B}{\|A\| \|B\|}$

## Similarity search vs tìm theo từ khóa
+ Similarity search: tìm theo ý nghĩa gần nhau (các vector tương đồng), có thể xử lý được từ đồng nghĩa và cách diễn đạt khác. Chi phí bao gồm không gian vector và chi phí so sánh
+ Tìm theo từ khóa: như cái tên. Chi phí rẻ, tốc độ nhanh

## Pipeline RAG
