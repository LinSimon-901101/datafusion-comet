-- Licensed to the Apache Software Foundation (ASF) under one
-- or more contributor license agreements.  See the NOTICE file
-- distributed with this work for additional information
-- regarding copyright ownership.  The ASF licenses this file
-- to you under the Apache License, Version 2.0 (the
-- "License"); you may not use this file except in compliance
-- with the License.  You may obtain a copy of the License at
--
--   http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing,
-- software distributed under the License is distributed on an
-- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
-- KIND, either express or implied.  See the License for the
-- specific language governing permissions and limitations
-- under the License.

-- Config: spark.sql.adaptive.enabled=false
-- Config: spark.sql.execution.reuseSubquery=false
-- ConstantFolding: enabled

statement
CREATE TABLE scalar_padding(id INT, s STRING, n INT) USING parquet

statement
INSERT INTO scalar_padding SELECT /*+ COALESCE(1) */ * FROM VALUES
  (1, 'ab', 5), (2, 'cd', 3) AS rows(id, s, n)

-- Keep default constant folding and use plain subqueries, independent of struct support.
-- Both rows take the scalar branch, so references do not imply an array at runtime.
query expect_native(lpad,rpad)
SELECT id, lpad(IF(id > 0, (SELECT max(s) FROM scalar_padding), s), 5, '0'),
       rpad(IF(id > 0, (SELECT max(s) FROM scalar_padding), s), 5, '0')
FROM scalar_padding

query expect_native(lpad,rpad)
SELECT id, lpad(CASE WHEN id > 0 THEN (SELECT max(s) FROM scalar_padding) ELSE s END, 5, '0'),
       rpad(CASE WHEN id > 0 THEN (SELECT max(s) FROM scalar_padding) ELSE s END, 5, '0')
FROM scalar_padding

-- Selecting the ELSE branch for the entire batch must preserve the scalar result too.
query expect_native(lpad,rpad)
SELECT id, lpad(IF(id < 0, s, (SELECT max(s) FROM scalar_padding)), 5, '0'),
       rpad(IF(id < 0, s, (SELECT max(s) FROM scalar_padding)), 5, '0')
FROM scalar_padding

-- A plain scalar string and scalar length stay scalar until the projection broadcasts them.
query expect_native(lpad,rpad)
SELECT id, lpad((SELECT max(s) FROM scalar_padding), 5, '0'),
       rpad((SELECT max(s) FROM scalar_padding), 5, '0')
FROM scalar_padding

-- Row-dependent lengths produce arrays from the same scalar string.
query expect_native(lpad,rpad)
SELECT id, lpad(IF(id > 0, (SELECT max(s) FROM scalar_padding), s), n, '0'),
       rpad(IF(id > 0, (SELECT max(s) FROM scalar_padding), s), n, '0')
FROM scalar_padding

-- Mixed rows also use the existing array kernel.
query expect_native(lpad,rpad)
SELECT id, lpad(IF(id = 1, (SELECT max(s) FROM scalar_padding), s), n, '0'),
       rpad(IF(id = 1, (SELECT max(s) FROM scalar_padding), s), n, '0')
FROM scalar_padding

-- Empty subqueries yield typed scalar NULLs. NULL scalar lengths preserve the output shape.
query expect_native(lpad,rpad)
SELECT id, lpad(IF(id > 0, (SELECT max(s) FROM scalar_padding WHERE id < 0), s), 5, '0'),
       rpad(IF(id > 0, (SELECT max(s) FROM scalar_padding WHERE id < 0), s), n, '0')
FROM scalar_padding

query expect_native(lpad,rpad)
SELECT id, lpad(s, (SELECT max(n) FROM scalar_padding WHERE id < 0), '0'),
       rpad(s, (SELECT max(n) FROM scalar_padding WHERE id < 0), '0')
FROM scalar_padding

query expect_native(lpad,rpad)
SELECT id, lpad(IF(id > 0, (SELECT max(s) FROM scalar_padding), s),
                (SELECT max(n) FROM scalar_padding WHERE id < 0), '0'),
       rpad(IF(id > 0, (SELECT max(s) FROM scalar_padding), s),
                (SELECT max(n) FROM scalar_padding WHERE id < 0), '0')
FROM scalar_padding

-- Unicode padding and truncation count characters, including multibyte scalars.
query expect_native(lpad,rpad)
SELECT id, lpad(IF(id > 0, 'é🙂', s), 5, '界a'),
       rpad(IF(id > 0, 'é🙂', s), 5, '界a'),
       lpad(IF(id > 0, 'é🙂', s), 1, '界a'),
       rpad(IF(id > 0, 'é🙂', s), 1, '界a')
FROM scalar_padding
