from sentence_transformers import SentenceTransformer
from pymilvus import connections, Collection

connections.connect("default", host="localhost", port="19530")
collection = Collection("demo_collection")
collection.load()

query = "Lưu trữ vector cho AI"
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode([query]).tolist()

results = collection.search(embedding, "embedding", {"metric_type": "L2", "params": {"nprobe": 10}}, limit=3)
for result in results[0]:
    print(f"🎯 ID: {result.id}, Distance: {result.distance}")
