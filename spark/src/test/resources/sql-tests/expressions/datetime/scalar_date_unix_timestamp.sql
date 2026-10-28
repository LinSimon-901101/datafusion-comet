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

-- ConfigMatrix: spark.sql.session.timeZone=UTC,America/Los_Angeles
-- Config: spark.comet.exec.scalaUDF.codegen.enabled=true
-- Config: spark.sql.adaptive.enabled=false
-- Config: spark.sql.execution.reuseSubquery=false

statement
CREATE TABLE scalar_date_unix_timestamp(id INT, d DATE) USING parquet

statement
INSERT INTO scalar_date_unix_timestamp VALUES (1, DATE '2024-01-15'), (2, NULL)

query expect_native(unix_timestamp)
SELECT unix_timestamp(d) FROM scalar_date_unix_timestamp

-- Java applies recurring summer DST rules in 2200; chrono-tz's last offset is insufficient.
query expect_dispatch(unix_timestamp)
SELECT unix_timestamp(IF(id > 0, DATE '2200-07-01', d)) FROM scalar_date_unix_timestamp

-- Spark's daysToMicros uses checked multiplication, including when ANSI is disabled.
query expect_error(long overflow)
SELECT unix_timestamp(IF(id > 0, date_from_unix_date(2147483647), d))
FROM scalar_date_unix_timestamp

statement
SET spark.comet.exec.scalaUDF.codegen.enabled=false

query expect_fallback(unix_timestamp date inputs that may be scalar require Spark for checked timezone conversion)
SELECT unix_timestamp(IF(id > 0, DATE '2200-07-01', d)) FROM scalar_date_unix_timestamp


statement
CREATE TABLE date_array_inputs(id INT, p BOOLEAN, d1 DATE, d2 DATE) USING parquet

statement
INSERT INTO date_array_inputs SELECT /*+ COALESCE(1) */ * FROM VALUES
  (1, true, DATE '2024-07-01', DATE '2024-01-01'),
  (2, true, CAST(NULL AS DATE), DATE '2024-01-02'),
  (3, false, DATE '2024-07-03', DATE '2024-01-03'),
  (4, false, CAST(NULL AS DATE), CAST(NULL AS DATE)) AS rows(id, p, d1, d2)

-- Column-only conditional results retain the existing native Date32 array kernel.
query expect_native(unix_timestamp)
SELECT id, unix_timestamp(IF(p, d1, d2)) FROM date_array_inputs

query expect_native(unix_timestamp)
SELECT id, unix_timestamp(IF(p, d1, d2)) FROM date_array_inputs WHERE id < 3

query expect_native(unix_timestamp)
SELECT id, unix_timestamp(coalesce(d1, d2)) FROM date_array_inputs

query expect_native(unix_timestamp)
SELECT id, unix_timestamp(CASE WHEN p THEN d1 ELSE d2 END) FROM date_array_inputs

query expect_native(unix_timestamp)
SELECT id, unix_timestamp(IF(p, coalesce(d1, d2), CASE WHEN id = 3 THEN d1 ELSE d2 END))
FROM date_array_inputs

-- These shapes can still return scalar dates and must retain checked Spark conversion.
query expect_fallback(unix_timestamp date inputs that may be scalar require Spark for checked timezone conversion)
SELECT unix_timestamp(IF(p, d1, DATE '2200-07-01')) FROM date_array_inputs

query expect_fallback(unix_timestamp date inputs that may be scalar require Spark for checked timezone conversion)
SELECT unix_timestamp(CASE WHEN p THEN d1 ELSE (SELECT max(d2) FROM date_array_inputs) END)
FROM date_array_inputs

query expect_fallback(unix_timestamp date inputs that may be scalar require Spark for checked timezone conversion)
SELECT unix_timestamp(coalesce(d1, DATE '2200-07-01')) FROM date_array_inputs

query expect_fallback(unix_timestamp date inputs that may be scalar require Spark for checked timezone conversion)
SELECT unix_timestamp(CASE WHEN id < 0 THEN d1 END) FROM date_array_inputs
