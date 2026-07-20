import psycopg2, yaml

with open('config.yaml', 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

db = cfg['database']
conn = psycopg2.connect(
    host=db['host'], port=db['port'],
    dbname=db['database'], user=db['user'], password=db['password']
)
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM category_vectors')
total = cur.fetchone()[0]

cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='category_vectors'")
cols = [r[0] for r in cur.fetchall()]
print(f'category_vectors: {total} rows, columns: {cols}')

if 'vec_search' in cols:
    cur.execute('SELECT COUNT(*) FROM category_vectors WHERE vec_search IS NOT NULL')
    vs = cur.fetchone()[0]
    print(f'vec_search non-null: {vs}/{total}')
else:
    print('vec_search column NOT exists')

cur.execute("SELECT indexname FROM pg_indexes WHERE tablename='category_vectors'")
indexes = [r[0] for r in cur.fetchall()]
print(f'Indexes: {indexes}')

cur.execute('SELECT COUNT(*) FROM category_vectors WHERE embedding IS NOT NULL')
emb = cur.fetchone()[0]
print(f'embedding non-null: {emb}/{total}')

cur.execute("""
    SELECT ct.category_id, ct.category_name 
    FROM category_texts ct 
    WHERE NOT EXISTS (SELECT 1 FROM category_vectors cv WHERE cv.category_id = ct.category_id)
    LIMIT 5
""")
missing = cur.fetchall()
if missing:
    print(f'Missing from vectors ({len(missing)} shown):')
    for m in missing:
        print(f'  {m[0]}: {m[1]}')

conn.close()
