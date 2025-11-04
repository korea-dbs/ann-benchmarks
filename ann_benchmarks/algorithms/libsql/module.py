import sqlite3
import numpy as np
import os
import sys
import time
import struct 
from ..base.module import BaseANN

class LibSQL(BaseANN):
    """
    Configure LD_LIBRARY_PATH
       export LD_LIBRARY_PATH=/path/to/libsql-sqlite3/.libs:$LD_LIBRARY_PATH
    #   algo = LibSQL('angular', {'use_index': True})
    """
    
    def __init__(self, metric, use_index=True, max_neighbors=None, compress_neighbors=None, distance_metric=None, db_path=":memory:"):
        """
        db_path: memory (default)
        """
        self._metric = metric
        self._use_index = use_index
        self._distance_metric = distance_metric or ("cosine" if metric == "angular" else "l2")
        self._max_neighbors = max_neighbors
        self._compress_neighbors = compress_neighbors
        self._db_path = db_path
        
        self._sainty_check()
        
        # database connection
        self.conn = sqlite3.connect(self._db_path)
        self.cursor = self.conn.cursor()
        
        self._dimension = None
        self._table_name = "vectors"
        self._index_name = "vectors_idx"
        self.name = self._build_name()
        
        print(f"Connected to libSQL (SQLite {sqlite3.sqlite_version})")
        print(f"Database: {self._db_path}")

    def _build_name(self):
        if self._use_index:
            parts = [f"diskann"]
            if self._max_neighbors:
                parts.append(f"n{self._max_neighbors}")
            if self._compress_neighbors:
                parts.append(f"c{self._compress_neighbors}")
            return f"LibSQL({'-'.join(parts)},{self._distance_metric})"
        else:
            return f"LibSQL(bruteforce,{self._distance_metric})"
    
    def _sainty_check(self):
        try:
            test_conn = sqlite3.connect(":memory:")
            test_cursor = test_conn.cursor()
            
            test_cursor.execute("SELECT vector('[1,2,3]')")
            
            test_conn.close()
            print("libSQL works!")
            
        except sqlite3.OperationalError as e:
            print(f"WARNING: libSQL vector functions not found!")
            raise RuntimeError("libSQL vector support not available. "
                             "Build libsql-sqlite3 and set LD_LIBRARY_PATH correctly.")
    
    def fit(self, X):
        """
        Construct baseline table
        """

        self._dimension = X.shape[1]
        n_vectors = X.shape[0]
        
        print(f"\nFitting {n_vectors} vectors of dimension {self._dimension}")
        print(f"Use index: {self._use_index}, Metric: {self._distance_metric}")
        
        try:
            self.cursor.execute(f"DROP TABLE IF EXISTS {self._table_name}")
            self.conn.commit()
        except Exception as e:
            print(f"Note: {e}")
       

        """
        Current version stores vector embedding into F32_BLOB data type.
        """ 
        create_table_sql = f"""
            CREATE TABLE {self._table_name} (
                id INTEGER PRIMARY KEY,
                embedding F32_BLOB({self._dimension})
            )
        """
        
        print("Creating table...")
        self.cursor.execute(create_table_sql)
        self.conn.commit()
        
        print("Inserting vectors...")
        batch_size = 1000
        
        import time
        insert_start = time.time()
        
        for i in range(0, len(X), batch_size):
            batch = X[i:i+batch_size]
            
            self.cursor.execute("BEGIN TRANSACTION")
            
            try:
                for idx, vec in enumerate(batch):
                    vec_id = i + idx
                    vec_str = '[' + ','.join(f"{v:.6f}" for v in vec) + ']'
                    
                    self.cursor.execute(
                        f"INSERT INTO {self._table_name} (id, embedding) VALUES (?, vector(?))",
                        (vec_id, vec_str)
                    )
                
                self.conn.commit()
                
                if (i + batch_size) % 10000 == 0:
                    elapsed = time.time() - insert_start
                    rate = (i + batch_size) / elapsed
                    print(f"  Inserted {min(i + batch_size, len(X))}/{len(X)} "
                          f"({rate:.0f} vec/s)")
                    
            except Exception as e:
                self.conn.rollback()
                print(f"Error at batch {i}: {e}")
                raise
        
        insert_time = time.time() - insert_start
        print(f"[INFO] Insertion complete ({insert_time:.2f}s, {len(X)/insert_time:.0f} vec/s)")
        
        # Construct diskAnn index
        if self._use_index:
            print("\nCreating DiskANN index...")
            params = []
            
            if self._distance_metric == "l2":
                params.append("'metric=l2'")
            
            if self._max_neighbors is not None:
                params.append(f"'max_neighbors={self._max_neighbors}'")
            
            if self._compress_neighbors is not None:
                params.append(f"'compress_neighbors={self._compress_neighbors}'")
            
            if params:
                params_str = ', '.join(params)
                create_index_sql = f"""
                    CREATE INDEX {self._index_name} ON {self._table_name} (
                        libsql_vector_idx(embedding, {params_str})
                    )
                """
            else:
                create_index_sql = f"""
                    CREATE INDEX {self._index_name} ON {self._table_name} (
                        libsql_vector_idx(embedding)
                    )
                """
            
            print(f"Index params: {params_str if params else 'default'}")
            
            index_start = time.time()
            self.cursor.execute(create_index_sql)
            self.conn.commit()
            index_time = time.time() - index_start
            
            print(f"[INFO] Index created ({index_time:.2f}s)")
        else:
            print("Skipping index (brute-force mode)")
        
        print("[INFO] Fit complete!\n")
    
    def query(self, v, n):
        """
        KNN 검색
        """
        #print("query call")
        vec_str = '[' + ','.join(f"{x:.6f}" for x in v) + ']'
        
        if self._use_index:
            result = self._query_with_index(vec_str, n)
        else:
            result = self._query_bruteforce(vec_str, n)
        
        return np.array(result, dtype=np.int32)


    def batch_query(self, X, n):
        print(f"\n{'='*60}")
        print(f"Starting batch query: {len(X):,} queries")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        results = []
        query_times = []
        
        for i, v in enumerate(X):
            query_start = time.time()
            results.append(self.query(v, n))
            query_times.append(time.time() - query_start)
            
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i+1:,}/{len(X):,} queries...")
        
        total_time = time.time() - start_time
        query_times = np.array(query_times)
        
        print(f"\n{'='*60}")
        print(f"QUERY PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        print(f"Total queries:        {len(X):,}")
        print(f"Total time:           {total_time:.2f}s")
        print(f"QPS:                  {len(X)/total_time:.2f} queries/sec")
        print(f"Avg query time:       {np.mean(query_times)*1000:.2f}ms")
        print(f"Median query time:    {np.median(query_times)*1000:.2f}ms")
        print(f"P95 query time:       {np.percentile(query_times, 95)*1000:.2f}ms")
        print(f"P99 query time:       {np.percentile(query_times, 99)*1000:.2f}ms")
        print(f"Min query time:       {np.min(query_times)*1000:.2f}ms")
        print(f"Max query time:       {np.max(query_times)*1000:.2f}ms")
        print(f"{'='*60}\n")
        
        return results

    def _query_with_index(self, vec_str, n):
    
        query_sql = f"""
        SELECT v.id
        FROM vector_top_k('{self._index_name}', vector(?), ?) AS vt
        JOIN {self._table_name} AS v ON v.rowid = vt.rowid
        """
    
        try:
            self.cursor.execute(query_sql, (vec_str, n))
            results = self.cursor.fetchall()
            return [int(row[0]) for row in results]
        except Exception as e:
            print(f"Index query error: {e}, falling back to brute-force")
            return self._query_bruteforce(vec_str, n)

   
    def _query_bruteforce(self, vec_str, n):
        """
        Full table scan
        """
        if self._distance_metric == "cosine" or self._metric == "angular":
            distance_func = "vector_distance_cos"
        else:
            distance_func = "vector_distance_l2"
        
        query_sql = f"""
            SELECT id
            FROM {self._table_name}
            ORDER BY {distance_func}(embedding, vector(?))
            LIMIT ?
        """
        
        self.cursor.execute(query_sql, (vec_str, n))
        results = self.cursor.fetchall()
        return [int(row[0]) for row in results]
    
    def __str__(self):
        if self._use_index:
            parts = [f"index,{self._distance_metric}"]
            if self._max_neighbors:
                parts.append(f"n={self._max_neighbors}")
            if self._compress_neighbors:
                parts.append(f"c={self._compress_neighbors}")
            return f"LibSQL({','.join(parts)})"
        else:
            return f"LibSQL(bruteforce,{self._distance_metric})"
    
    def __del__(self):
        if hasattr(self, 'cursor'):
            self.cursor.close()
        if hasattr(self, 'conn'):
            self.conn.close()
