from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('all-mpnet-base-v2')

@app.route('/api/encode', methods=['POST'])
def encode():
    try:
        # 从请求中获取文本数据
        text = request.get_json().get('text')

        # 使用模型将文本编码为向量
        vector = model.encode(text)

        # 将向量转换为JSON格式并返回
        return jsonify({'vector': vector.tolist()})

    except Exception as e:
        # 返回错误消息和状态码500服务器错误
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
