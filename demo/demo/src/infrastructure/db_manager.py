from __future__ import annotations
import logging
from typing import Any

import psycopg2
from psycopg2 import pool

from src.models.config_models import DBConfig


class DBConnectionManager:
    def __init__(self, config: DBConfig):
        self._config = config
        self._pool: pool.SimpleConnectionPool | None = None
        self._logger = logging.getLogger("DBConnectionManager")

    def initialize(self) -> None:
        try:
            self._pool = pool.SimpleConnectionPool(
                minconn=1,
                maxconn=self._config.pool_size,
                host=self._config.host,
                port=self._config.port,
                dbname=self._config.database,
                user=self._config.user,
                password=self._config.password,
            )
            self._logger.info("数据库连接池初始化成功")
        except Exception as e:
            self._logger.error(f"数据库连接池初始化失败: {e}")
            raise

    def get_connection(self):
        if self._pool is None:
            raise RuntimeError("数据库连接池未初始化，请先调用 initialize()")
        return self._pool.getconn()

    def put_connection(self, conn) -> None:
        if self._pool is not None:
            self._pool.putconn(conn)

    def execute(self, query: str, params: tuple | None = None) -> list[dict[str, Any]]:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                if cur.description is None:
                    conn.commit()
                    return []
                columns = [desc[0] for desc in cur.description]
                rows = cur.fetchall()
                conn.commit()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            conn.rollback()
            self._logger.error(f"SQL执行失败: {e}, query={query[:100]}")
            raise
        finally:
            self.put_connection(conn)

    def execute_one(self, query: str, params: tuple | None = None) -> dict[str, Any] | None:
        results = self.execute(query, params)
        return results[0] if results else None

    def execute_batch(self, query: str, params_list: list[tuple]) -> int:
        conn = self.get_connection()
        try:
            with conn.cursor() as cur:
                cur.executemany(query, params_list)
                affected = cur.rowcount
                conn.commit()
                return affected
        except Exception as e:
            conn.rollback()
            self._logger.error(f"批量SQL执行失败: {e}, query={query[:100]}")
            raise
        finally:
            self.put_connection(conn)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.closeall()
            self._pool = None
            self._logger.info("数据库连接池已关闭")
