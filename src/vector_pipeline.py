from pyspark.sql import SparkSession
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection

# 1. Init
spark = SparkSession.builder.appName("MilvusDemo").getOrCreate()
model = SentenceTransformer("all-MiniLM-L6-v2")
connections.connect("default", host="localhost", port="19530")

# 2. Đọc dữ liệu
df = spark.read.text("../data/sample_sentences.txt").toDF("text").withColumn("id", spark.sql.functions.monotonically_increasing_id())
texts = [row["text"] for row in df.collect()]
embeddings = model.encode(texts).tolist()
ids = [int(row["id"]) for row in df.collect()]

# 3. Tạo collection
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
]
schema = CollectionSchema(fields, description="Demo collection")
collection = Collection(name="demo_collection", schema=schema)
collection.insert([ids, embeddings])
collection.flush()
print("✅ Đã insert vào Milvus.")
