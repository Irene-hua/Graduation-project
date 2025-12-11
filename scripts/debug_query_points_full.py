from qdrant_client import QdrantClient

c = QdrantClient(path='D:/PycharmProjects/Graduation-project/qdrant_storage')
print('Client type:', type(c))

# This debug helper was used during development and has been removed to keep the repository clean.
