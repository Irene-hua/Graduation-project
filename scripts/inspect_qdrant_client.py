from qdrant_client import QdrantClient
import inspect

c = QdrantClient(path='D:/PycharmProjects/Graduation-project/qdrant_storage')
print('QdrantClient:', type(c))

names = dir(c)
print('\nMethods/properties related to search/upsert/get:')
for a in names:
    low = a.lower()
    if any(k in low for k in ('search','find','query','point','collection','upsert','get','scroll')):
        print(' -', a)

print('\nChecking signatures for common names:')
for name in ['search','upsert','get_collection','scroll','get_point','get_points','retrieve','search_points','search_collection','search_batch']:
    if hasattr(c, name):
        try:
            sig = inspect.signature(getattr(c, name))
        except Exception as e:
            sig = f'<signature error: {e}>'
        print(f'Found {name}: {sig}')

# Try to introspect where vectors might be stored (if sqlite local storage)
print('\nClient repr:', repr(c))

# Try minimal safe call: get_collections / get_collection (no mutation)
for nm in ('get_collections','get_collection'):
    if hasattr(c, nm):
        try:
            print(f'Calling {nm}() ->')
            res = getattr(c, nm)()
            print(type(res), getattr(res, '__dict__', str(res))[:200])
        except Exception as e:
            print(f'{nm}() failed: {e}')

print('\nDone')

