from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility
from langchain_together import TogetherEmbeddings
from config import ZILLIZ_CLOUD_URI, ZILLIZ_CLOUD_API_KEY

def create_vector_store(chunks: list[str]):
    connections.connect("default", uri=ZILLIZ_CLOUD_URI, token=ZILLIZ_CLOUD_API_KEY)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535)
    ]
    schema = CollectionSchema(fields, "PDF Document Chunks")

    collection_name = "clean_pdf_chunks"
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
    else:
        collection = Collection(collection_name, schema)

    embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
    chunk_embeddings = embeddings.embed_documents(chunks)

    entities = [
        chunk_embeddings,
        chunks
    ]

    collection.insert(entities)
    collection.create_index("embedding", {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 128}})
    collection.load()

    return collection

def query_zilliz_database(query: str, k: int = 3) -> list[str]:
    try:
        connections.connect("default", uri=ZILLIZ_CLOUD_URI, token=ZILLIZ_CLOUD_API_KEY)
        collection_name = "clean_pdf_chunks"
        if not utility.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist.")
        
        collection = Collection(collection_name)
        collection.load()

        embeddings = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")
        query_embedding = embeddings.embed_query(query)

        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        results = collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=k,
            output_fields=["text"]
        )

        retrieved_documents = []
        for hits in results:
            for hit in hits:
                retrieved_documents.append(hit.entity.get("text"))

        return retrieved_documents

    except Exception as e:
        print(f"Error querying Zilliz database: {e}")
        raise